import assert from "node:assert/strict";
import test from "node:test";

import { executeSearchPipeline } from "../../src/pi-search/search_pipeline";
import type { PiSearchBackend } from "../../src/pi-search/searcher/contract/interface";
import type { SearchBackendSearchResponse } from "../../src/pi-search/searcher/contract/types";
import type { RerankScorer } from "../../src/pi-search/reranking/types";

const identityScorer: RerankScorer = {
  score: async (_criterion, candidates) =>
    candidates.map((candidate, index) => ({
      ...candidate,
      rerankScore: candidates.length - index,
    })),
};

function searchableBackend(): PiSearchBackend {
  return {
    ...createBackend(async () => ({
      hits: [
        { docid: "a", score: 10, title: "A", snippet: "Preview A" },
        { docid: "b", score: 8, title: "B", snippet: "Preview B" },
        { docid: "a", score: 12 },
      ],
      hasMore: true,
      nextOffset: 4,
      totalHits: 20,
      timingMs: { request: 12 },
    })),
    readDocument: async ({ docid }) => ({
      found: true,
      docid,
      text: docid.repeat(5000),
      offset: 1,
      limit: 1,
      truncated: false,
    }),
  };
}

function createBackend(search: PiSearchBackend["search"]): PiSearchBackend {
  return {
    capabilities: {
      backendId: "mock",
      supportsScore: true,
      supportsSnippets: true,
      supportsExactTotalHits: true,
    },
    search,
    readDocument: async () => {
      throw new Error("pipeline must not read documents yet");
    },
  };
}

void test("search pipeline forwards request and signal and preserves the complete response", async () => {
  const request = { query: "analytical engine", limit: 2 };
  const signal = new AbortController().signal;
  const response: SearchBackendSearchResponse = {
    hits: [
      { docid: "doc-2", score: 3, title: "Babbage", snippet: "Engine design" },
      { docid: "doc-1", score: 1, metadata: { source: "test" } },
    ],
    totalHits: 10,
    hasMore: true,
    nextOffset: 3,
    timingMs: { request: 12, backendInit: 2, backendUptime: 100 },
  };
  let calls = 0;
  const backend = createBackend(async (actualRequest, actualSignal) => {
    calls += 1;
    assert.strictEqual(actualRequest, request);
    assert.strictEqual(actualSignal, signal);
    return response;
  });

  assert.strictEqual(await executeSearchPipeline(backend, request, signal), response);
  assert.equal(calls, 1);
});

void test("reranked search retrieves a larger pool, reads unique documents, and returns requested hits", async () => {
  const backend = searchableBackend();
  const search = backend.search.bind(backend);
  const read = backend.readDocument.bind(backend);
  const signal = new AbortController().signal;
  const readDocids: string[] = [];
  backend.search = async (request, receivedSignal) => {
    assert.deepEqual(request, { query: "engine", limit: 50 });
    assert.strictEqual(receivedSignal, signal);
    return search(request, receivedSignal);
  };
  backend.readDocument = async (request, receivedSignal) => {
    readDocids.push(request.docid);
    assert.strictEqual(receivedSignal, signal);
    return read(request, receivedSignal);
  };
  const response = await executeSearchPipeline(backend, { query: "engine", limit: 1 }, signal, {
    mode: "fixed-template",
    candidateK: 50,
    scorer: {
      score: async (criterion, candidates, receivedSignal) => {
        assert.strictEqual(receivedSignal, signal);
        assert.equal(criterion.question, "engine");
        assert.deepEqual(
          candidates.map((candidate) => candidate.text.length),
          [4000, 4000],
        );
        assert.equal(candidates[0].title, "A");
        assert.equal(candidates[1].title, "B");
        return candidates.map((candidate, index) => ({ ...candidate, rerankScore: index }));
      },
    },
  });
  assert.deepEqual(readDocids, ["a", "b"]);
  assert.deepEqual(response.hits, [
    { docid: "b", score: 8, title: "B", snippet: "Preview B", rerankScore: 1 },
  ]);
  assert.deepEqual(response.reranking?.candidateDocids, ["a", "b"]);
  assert.deepEqual(response.reranking?.ranking, [
    { docid: "b", retrievalScore: 8, rerankScore: 1 },
    { docid: "a", retrievalScore: 10, rerankScore: 0 },
  ]);
  assert.equal(response.totalHits, 20);
  assert.deepEqual(response.timingMs, { request: 12 });
  assert.equal(response.hasMore, false);
  assert.equal(response.nextOffset, undefined);
});

void test("retrieval depth is at least requested hits, with no padding when candidates are scarce", async () => {
  const backend = searchableBackend();
  const search = backend.search.bind(backend);
  backend.search = async (request, signal) => {
    assert.equal(request.limit, 5);
    return search(request, signal);
  };
  const response = await executeSearchPipeline(backend, { query: "engine", limit: 5 }, undefined, {
    mode: "fixed-template",
    candidateK: 1,
    scorer: identityScorer,
  });
  assert.equal(response.hits.length, 2);
});

void test("customized search uses the actual search query and generates one question", async () => {
  let calls = 0;
  const response = await executeSearchPipeline(
    searchableBackend(),
    { query: "current query", limit: 2 },
    undefined,
    {
      mode: "customized",
      candidateK: 5,
      originalQuestion: "Original task",
      searchReason: "Find evidence",
      generator: {
        generateQuestion: async (context) => {
          calls += 1;
          assert.deepEqual(context, {
            currentQuery: "current query",
            originalQuestion: "Original task",
            searchReason: "Find evidence",
          });
          return "Does this document identify the designer?";
        },
      },
      scorer: identityScorer,
    },
  );
  assert.equal(calls, 1);
  assert.equal(
    response.reranking?.criterion?.question,
    "Does this document identify the designer?",
  );
});

void test("empty retrieval skips document reads, generation, and scoring", async () => {
  const backend = createBackend(async () => ({ hits: [], hasMore: false }));
  const response = await executeSearchPipeline(backend, { query: "engine", limit: 5 }, undefined, {
    mode: "customized",
    candidateK: 10,
    originalQuestion: "Task",
    generator: {
      generateQuestion: async () => {
        throw new Error("must not generate");
      },
    },
    scorer: {
      score: async () => {
        throw new Error("must not score");
      },
    },
  });
  assert.deepEqual(response.hits, []);
  assert.deepEqual(response.reranking, { criterion: null, candidateDocids: [], ranking: [] });
});

void test("rerank failures propagate instead of returning original search results", async () => {
  const error = new Error("reranker unavailable");
  await assert.rejects(
    executeSearchPipeline(searchableBackend(), { query: "engine", limit: 1 }, undefined, {
      mode: "fixed-template",
      candidateK: 5,
      scorer: {
        score: async () => {
          throw error;
        },
      },
    }),
    (actual) => actual === error,
  );
});

void test("reranked search validates candidate depth before searching", async () => {
  const backend = createBackend(async () => {
    throw new Error("must not search");
  });
  for (const candidateK of [0, -1, 1.5]) {
    await assert.rejects(
      executeSearchPipeline(backend, { query: "engine", limit: 1 }, undefined, {
        mode: "fixed-template",
        candidateK,
        scorer: identityScorer,
      }),
      /candidateK must be a positive integer/,
    );
  }
});

void test("canceled reranked search makes no backend request", async () => {
  const backend = createBackend(async () => {
    throw new Error("must not search");
  });
  await assert.rejects(
    executeSearchPipeline(backend, { query: "engine", limit: 1 }, AbortSignal.abort(), {
      mode: "fixed-template",
      candidateK: 5,
      scorer: identityScorer,
    }),
    { name: "AbortError" },
  );
});

void test("search pipeline propagates backend failures unchanged", async () => {
  const error = new Error("backend unavailable");
  const backend = createBackend(async () => {
    throw error;
  });

  await assert.rejects(
    executeSearchPipeline(backend, { query: "test", limit: 5 }),
    (actual) => actual === error,
  );
});
