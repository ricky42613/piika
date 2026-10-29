import assert from "node:assert/strict";
import test from "node:test";

import { executeSearchPipeline } from "../../src/pi-search/search_pipeline";
import type { PiSearchBackend } from "../../src/pi-search/searcher/contract/interface";
import type { SearchBackendSearchResponse } from "../../src/pi-search/searcher/contract/types";

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
