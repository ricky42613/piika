import assert from "node:assert/strict";
import test from "node:test";
import { buildMockExtensionConfig } from "../../src/pi-search/config";
import { PiSearchBackendRuntime } from "../../src/pi-search/searcher/runtime";
import { SearchSessionStore } from "../../src/pi-search/search_cache";
import { ManagedTempSpillDir } from "../../src/pi-search/spill";
import { executeDirectSearchTool } from "../../src/pi-search/tool_handlers";
import { createDocumentReranker, isSearchRerankEnabled } from "../../src/pi-search/reranker";
import { filterScoreFromYesProbability } from "../../src/pi-search/yes_no_filter";
import { searchFilterAction } from "../../src/pi-search/filter_config";

test("rerank mode is opt-in and only applies to direct 2-tool search", () => {
  assert.equal(searchFilterAction({ PI_SEARCH_FILTER_ACTION: "rerank" }), "rerank");
  assert.equal(searchFilterAction({}), "filter");
  assert.throws(() => searchFilterAction({ PI_SEARCH_FILTER_ACTION: "wrong" }), /must be filter or rerank/);
  assert.equal(isSearchRerankEnabled("pyserini-rest-2tool", { PI_SEARCH_RERANK_ENABLED: "1" }), true);
  assert.equal(isSearchRerankEnabled("pi-serini-3tool", { PI_SEARCH_RERANK_ENABLED: "1" }), false);
  assert.equal(isSearchRerankEnabled("pyserini-rest-2tool", {}), false);
});

test("Qwen3 /score receives search query and returns an activated probability; rejects invalid scores", async () => {
  let body: Record<string, unknown> = {};
  const request = async (_url: string | URL | Request, init?: RequestInit) => {
    body = JSON.parse(init?.body as string);
    return new Response(JSON.stringify({ data: [{ index: 0, score: 0.8 }] }), { status: 200 });
  };
  const rerank = createDocumentReranker({ PI_SEARCH_RERANK_BASE_URL: "http://localhost:8152/v1", PI_SEARCH_RERANK_MODEL: "qwen3-reranker-4b" }, request as typeof fetch);
  assert.equal(await rerank("a search query", "doc"), 0.8);
  assert.equal(body.queries, "a search query");
  assert.deepEqual(body.documents, ["doc"]);
  assert.equal(body.use_activation, true);
  const invalid = createDocumentReranker({ PI_SEARCH_RERANK_BASE_URL: "http://localhost:8152/v1", PI_SEARCH_RERANK_MODEL: "model" },
    (async () => new Response(JSON.stringify({ data: [{ index: 0, score: NaN }] }), { status: 200 })) as typeof fetch);
  await assert.rejects(invalid("query", "doc"), /invalid score/);
});

test("direct search reranks top candidates without filtering, records original BM25 scores and preserves stable ties", async () => {
  const backendRuntime = new PiSearchBackendRuntime(buildMockExtensionConfig({ documents: [
    { docid: "one", text: "first" }, { docid: "two", text: "second" }, { docid: "three", text: "third" },
  ] }));
  const spillDir = new ManagedTempSpillDir("rerank-test-");
  const deps = {
    backendRuntime, searchStore: new SearchSessionStore(), spillDir,
    nextSpillSequence: () => 1, rerankDepth: 3,
    documentReranker: async (_query: string, document: string) => document === "third" ? 0.9 : 0.1,
  };
  try {
    const result = await executeDirectSearchTool({ query: "first second third", hits: 2, reason: "test" }, undefined, { cwd: "." }, deps);
    assert.deepEqual(result.details.retrievedDocids, ["three", "one"]);
    assert.equal(result.details.rerank?.candidateCount, 3);
    assert.deepEqual(result.details.rerank?.scores.map(s => s.docid), ["one", "three", "two"]);
    assert.ok(result.details.rerank?.scores.every(s => typeof s.bm25Score === "number"));
    assert.equal(result.details.filter, undefined);
    assert.match(result.content[0].text, /Qwen3 reranked ranking/);
    deps.documentReranker = async () => { throw new Error("scorer unavailable"); };
    await assert.rejects(executeDirectSearchTool({ query: "first second third", hits: 2, reason: "test" }, undefined, { cwd: "." }, deps), /scorer unavailable/);
  } finally {
    backendRuntime.dispose();
    spillDir.cleanup();
  }
});

test("Qorrect question scores rerank every candidate without threshold-based filtering", async () => {
  const backendRuntime = new PiSearchBackendRuntime(buildMockExtensionConfig({ documents: [
    { docid: "one", text: "first" }, { docid: "two", text: "second" }, { docid: "three", text: "third" },
  ] }));
  const spillDir = new ManagedTempSpillDir("qorrect-rerank-test-");
  let generated = 0;
  const deps = {
    backendRuntime, searchStore: new SearchSessionStore(), spillDir,
    nextSpillSequence: () => 1, rerankDepth: 3,
    filterAction: "rerank" as const, questionSource: "qorrect" as const, predictor: "qwen" as const,
    generateFilterQuestion: async (query: string, reason: string) => {
      assert.equal(query, "first second third");
      assert.equal(reason, "why");
      generated++;
      return "Does this document have relevant evidence?";
    },
    documentFilter: async (question: string, document: string) => {
      assert.equal(question, "Does this document have relevant evidence?");
      return filterScoreFromYesProbability(document === "second" ? 0.9 : document === "first" ? 0.2 : 0.1);
    },
  };
  try {
    const result = await executeDirectSearchTool({ query: "first second third", hits: 3, reason: "why" }, undefined, { cwd: "." }, deps);
    assert.equal(generated, 1);
    assert.deepEqual(result.details.retrievedDocids, ["two", "one", "three"]);
    assert.equal(result.details.rerank?.question, "Does this document have relevant evidence?");
    assert.equal(result.details.rerank?.candidateCount, 3);
    assert.equal(result.details.rerank?.yesNoScores?.filter(s => !s.keep).length, 2);
    assert.equal(result.details.filter, undefined);
  } finally {
    backendRuntime.dispose();
    spillDir.cleanup();
  }
});
