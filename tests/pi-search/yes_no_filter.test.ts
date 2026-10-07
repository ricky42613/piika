import assert from "node:assert/strict";
import test from "node:test";
import {
  createYesNoFilter,
  filterScoreFromYesProbability,
} from "../../src/pi-search/yes_no_filter";
import { registerPiSearchExtension } from "../../src/pi-search/extension";
import { buildMockExtensionConfig } from "../../src/pi-search/config";
import { PiSearchBackendRuntime } from "../../src/pi-search/searcher/runtime";
import { SearchSessionStore } from "../../src/pi-search/search_cache";
import { ManagedTempSpillDir } from "../../src/pi-search/spill";
import { executeDirectSearchTool } from "../../src/pi-search/tool_handlers";
import { formatPiSearchPrompt } from "../../src/pi-search/agent_prompt";

void test("strict Yes > No; normalized probabilities and JSON-safe logs", () => {
  assert.equal(filterScoreFromYesProbability(0.9).keep, true);
  assert.equal(filterScoreFromYesProbability(0.1).keep, false);
  assert.equal(filterScoreFromYesProbability(0.5).keep, false);
  assert.equal(filterScoreFromYesProbability(0).yesLogprob, null);
  assert.equal(filterScoreFromYesProbability(1).noLogprob, null);
  assert.equal(filterScoreFromYesProbability(0.8).yesLogprob, Math.log(0.8));
  for (const value of [NaN, Infinity, -0.1, 1.1])
    assert.throws(() => filterScoreFromYesProbability(value));
});

void test("filter errors propagate rather than returning unfiltered results", async () => {
  const filter = createYesNoFilter(
    { PI_SEARCH_FILTER_PREDICTOR: "qwen", PI_SEARCH_FILTER_BASE_URL: "http://qwen" },
    (async () => new Response("unavailable", { status: 503 })) as typeof fetch,
  );
  await assert.rejects(filter("q", "d"), /HTTP 503/);
});

void test("schema and prompts are gated to enabled two-tool mode", () => {
  const old = { ...process.env };
  try {
    process.env.PI_SEARCH_FILTER_PREDICTOR = "qwen";
    process.env.PI_SEARCH_FILTER_BASE_URL = "http://qwen";
    delete process.env.PI_SEARCH_FILTER_BACKEND;
    for (const enabled of ["0", "1"]) {
      process.env.PI_SEARCH_FILTER_ENABLED = enabled;
      for (const toolInterface of ["pyserini-rest-2tool", "pi-serini-3tool"] as const) {
        const tools: Array<{
          name: string;
          parameters: { required?: string[] };
          promptGuidelines?: string[];
        }> = [];
        registerPiSearchExtension(
          {
            on: () => {},
            registerTool: (tool: (typeof tools)[number]) => tools.push(tool),
          } as never,
          {
            toolInterface,
            resolveConfig: () => buildMockExtensionConfig({ documents: [] }),
          },
        );
        const expected = enabled === "1" && toolInterface === "pyserini-rest-2tool";
        assert.equal(tools[0].parameters.required?.includes("filter_question") ?? false, expected);
        assert.equal(tools[0].promptGuidelines?.join(" ").includes("filter_question"), expected);
        assert.equal(
          formatPiSearchPrompt("q", "plain_minimal", { toolInterface }).includes("filter_question"),
          expected,
        );
      }
    }
  } finally {
    for (const key of ["PI_SEARCH_FILTER_ENABLED", "PI_SEARCH_FILTER_PREDICTOR", "PI_SEARCH_FILTER_BASE_URL", "PI_SEARCH_FILTER_BACKEND"]) {
      if (old[key] === undefined) delete process.env[key];
      else process.env[key] = old[key];
    }
  }
});

void test("disabled filter and three-tool mode ignore invalid backend settings", () => {
  const oldEnabled = process.env.PI_SEARCH_FILTER_ENABLED;
  const oldBackend = process.env.PI_SEARCH_FILTER_BACKEND;
  try {
    process.env.PI_SEARCH_FILTER_BACKEND = "reranq";
    for (const [enabled, toolInterface] of [
      ["0", "pyserini-rest-2tool"],
      ["0", "pi-serini-3tool"],
      ["1", "pi-serini-3tool"],
    ] as const) {
      process.env.PI_SEARCH_FILTER_ENABLED = enabled;
      assert.doesNotThrow(() =>
        registerPiSearchExtension({ on: () => {}, registerTool: () => {} } as never, {
          toolInterface,
          resolveConfig: () => buildMockExtensionConfig({ documents: [] }),
        }),
      );
    }
    process.env.PI_SEARCH_FILTER_ENABLED = "1";
    assert.throws(
      () =>
        registerPiSearchExtension({ on: () => {}, registerTool: () => {} } as never, {
          toolInterface: "pyserini-rest-2tool",
          resolveConfig: () => buildMockExtensionConfig({ documents: [] }),
        }),
      /PI_SEARCH_FILTER_BACKEND is no longer supported/,
    );
  } finally {
    if (oldEnabled === undefined) delete process.env.PI_SEARCH_FILTER_ENABLED;
    else process.env.PI_SEARCH_FILTER_ENABLED = oldEnabled;
    if (oldBackend === undefined) delete process.env.PI_SEARCH_FILTER_BACKEND;
    else process.env.PI_SEARCH_FILTER_BACKEND = oldBackend;
  }
});

void test("direct search filters top REREANK_DEPTH candidates then returns requested hits", async () => {
  const spillDir = new ManagedTempSpillDir("rerank-depth-test-");
  const runtime = new PiSearchBackendRuntime(
    buildMockExtensionConfig({
      documents: [
        { docid: "a", text: "shared depth a", snippet: "preview" },
        { docid: "b", text: "shared depth b", snippet: "preview" },
        { docid: "c", text: "shared depth c", snippet: "preview" },
        { docid: "d", text: "shared depth d", snippet: "preview" },
        { docid: "e", text: "shared depth e", snippet: "preview" },
      ],
    }),
  );
  let scored = 0;
  const deps = {
    backendRuntime: runtime,
    searchStore: new SearchSessionStore(),
    spillDir,
    nextSpillSequence: () => 1,
    rerankDepth: 4,
    documentFilter: async () => {
      scored++;
      return filterScoreFromYesProbability(0.9);
    },
  };
  try {
    const result = await executeDirectSearchTool(
      { reason: "test", query: "shared depth", hits: 2, filter_question: "Does it match?" },
      undefined,
      { cwd: "." },
      deps,
    );
    assert.equal(scored, 4);
    assert.equal(result.details.filter?.candidateCount, 4);
    assert.equal(result.details.filter?.keptCount, 4);
    assert.equal(result.details.filter?.returnedCount, 2);
    assert.equal(result.details.filter?.requestedHits, 2);
    assert.equal(result.details.filter?.rereankDepth, 4);
    assert.deepEqual(result.details.retrievedDocids, ["a", "b"]);
    assert.match(result.content[0].text, /returning 2 of requested 2/);
  } finally {
    runtime.dispose();
    spillDir.cleanup();
  }
});

void test("direct search filters full documents, preserves ranking, records scores and validates question", async () => {
  const spillDir = new ManagedTempSpillDir("filter-test-");
  const runtime = new PiSearchBackendRuntime(
    buildMockExtensionConfig({
      documents: [
        { docid: "a", text: "shared clue\nfull evidence a", snippet: "preview" },
        { docid: "b", text: "shared clue\nfull evidence b", snippet: "preview" },
        { docid: "c", text: "shared clue\nfull evidence c", snippet: "preview" },
      ],
    }),
  );
  const seen: string[] = [];
  const deps = {
    backendRuntime: runtime,
    searchStore: new SearchSessionStore(),
    spillDir,
    nextSpillSequence: () => 1,
    documentFilter: async (question: string, document: string) => {
      assert.equal(question, "Does it contain evidence?");
      seen.push(document);
      return filterScoreFromYesProbability(document.includes("evidence b") ? 0.1 : 0.9);
    },
  };
  try {
    const params = {
      reason: "test",
      query: "shared",
      hits: 3,
      filter_question: "Does it contain evidence?",
    };
    const result = await executeDirectSearchTool(params, undefined, { cwd: "." }, deps);
    assert.deepEqual(result.details.retrievedDocids, ["a", "c"]);
    assert.equal(result.details.filter?.scores.length, 3);
    assert.equal(seen.length, 3);
    assert.ok(seen.every((text) => text.includes("full evidence")));
    assert.doesNotMatch(result.content[0].text, /docid=b/);
    await assert.rejects(
      executeDirectSearchTool({ ...params, filter_question: " " }, undefined, { cwd: "." }, deps),
      /filter_question/,
    );
    let generated = 0;
    let scored = 0;
    const generatedResult = await executeDirectSearchTool(
      { reason: "test", query: "shared", hits: 3, filter_question: "ignored" },
      undefined,
      { cwd: "." },
      {
        ...deps,
        generateFilterQuestion: async (query) => {
          generated++;
          assert.equal(query, "shared");
          return "Generated question?";
        },
        documentFilter: async (question) => {
          scored++;
          assert.equal(question, "Generated question?");
          return filterScoreFromYesProbability(0.9);
        },
      },
    );
    assert.equal(generated, 1);
    assert.equal(scored, 3);
    assert.equal(generatedResult.details.filter?.inputSource, "qorrect");
    assert.equal(generatedResult.details.filter?.question, "Generated question?");
    await assert.rejects(
      executeDirectSearchTool(
        params,
        undefined,
        { cwd: "." },
        {
          ...deps,
          generateFilterQuestion: async () => {
            throw new Error("generation failed");
          },
        },
      ),
      /generation failed/,
    );
    const relevance = await executeDirectSearchTool(
      { reason: "test", query: "shared", hits: 3 },
      undefined,
      { cwd: "." },
      {
        ...deps,
        questionSource: "general",
        documentFilter: async (query) => {
          assert.equal(query, "shared");
          return filterScoreFromYesProbability(0.9);
        },
      },
    );
    assert.equal(relevance.details.filter?.inputSource, "general");
    assert.equal(relevance.details.filter?.question, "shared");
    assert.deepEqual(relevance.details.retrievedDocids, ["a", "b", "c"]);
    const empty = await executeDirectSearchTool(
      params,
      undefined,
      { cwd: "." },
      { ...deps, documentFilter: async () => filterScoreFromYesProbability(0.5) },
    );
    assert.deepEqual(empty.details.retrievedDocids, []);
    assert.match(empty.content[0].text, /kept 0 of 3/);
  } finally {
    runtime.dispose();
    spillDir.cleanup();
  }
});
