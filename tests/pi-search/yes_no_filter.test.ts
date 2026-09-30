import assert from "node:assert/strict";
import test from "node:test";
import {
  CLASSIFICATION_SUFFIX,
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

function mockReranq(
  options: { score?: unknown; variants?: Record<string, number[]>; prefixLength?: number } = {},
) {
  const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
  const request = (async (url, init) => {
    assert.equal(typeof init?.body, "string");
    assert.equal(typeof url, "string");
    const body = JSON.parse(init!.body as string) as Record<string, unknown>;
    calls.push({ url: url as string, body });
    if ((url as string).endsWith("/tokenize")) {
      const variants = options.variants ?? { Yes: [1], " Yes": [2], No: [3], " No": [4] };
      const text = body.prompt as string;
      const tokens =
        variants[text] ??
        (text === CLASSIFICATION_SUFFIX
          ? [80, 81]
          : text.startsWith("Filter question:")
            ? Array(options.prefixLength ?? 3).fill(90)
            : Array.from({ length: 600 }, (_, i) => i + 100));
      return Response.json({ tokens });
    }
    assert.ok((url as string).endsWith("/generative_scoring"));
    return Response.json(
      options.score ?? {
        data: [{ index: 0, score: (body.label_token_ids as number[])[0] === 1 ? 0.6 : 0.2 }],
      },
    );
  }) as typeof fetch;
  return { calls, request };
}

void test("Reranq uses training prompt boundaries, first 512 document tokens and summed label variants", async () => {
  const { calls, request } = mockReranq();
  const filter = createYesNoFilter({}, request);
  const result = await filter("Is there evidence?", "document");
  assert.equal(result.yesProbability, 0.8);
  assert.ok(Math.abs(result.noProbability - 0.2) < 1e-10);
  assert.equal(result.keep, true);
  assert.equal(result.probabilitySemantics, "normalized_yes_no_variants");
  assert.equal(result.backend, "reranq");
  assert.equal(result.promptVersion, "reranq-training-v1");
  assert.equal(result.documentTokens, 512);
  assert.equal(result.documentTruncated, true);
  assert.equal(result.promptTokens, 517);
  assert.equal(result.scoringRequests, 2);
  assert.equal(calls[0].url, "http://127.0.0.1:8150/tokenize");
  assert.equal(calls[5].body.prompt, "Filter question:\nIs there evidence?\n\nDocument:\n");
  const scoring = calls.filter((c) => c.url.endsWith("/generative_scoring"));
  assert.deepEqual(
    scoring.map((c) => c.body.label_token_ids),
    [
      [1, 2, 3, 4],
      [2, 1, 3, 4],
    ],
  );
  assert.deepEqual(scoring[0].body.query, []);
  assert.equal(scoring[0].body.model, "reranq-filter");
  assert.equal(scoring[0].body.apply_softmax, true);
  assert.equal(scoring[0].body.add_special_tokens, false);
  const prompt = (scoring[0].body.items as number[][])[0];
  assert.deepEqual(prompt.slice(0, 3), [90, 90, 90]);
  assert.deepEqual(prompt.slice(-3), [611, 80, 81]);
  await filter("Another question?", "document");
  assert.equal(calls.length, 13); // Cache label variants and suffix, but not document/question.
  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(filter("q", "d", aborted.signal));
  assert.equal(calls.length, 13);
});

void test("Reranq rejects bad scores, incomplete responses and overlong questions", async () => {
  for (const score of [
    {},
    { data: [] },
    { data: [{ index: 1, score: 0.5 }] },
    { data: [{ index: 0, score: -1 }] },
    { data: [{ index: 0, score: 0.8 }] },
  ]) {
    await assert.rejects(createYesNoFilter({}, mockReranq({ score }).request)("q", "d"));
  }
  const { calls, request } = mockReranq({ prefixLength: 8190 });
  await assert.rejects(createYesNoFilter({}, request)("q", "d"), /too long/);
  assert.ok(calls.every((c) => !c.url.endsWith("/generative_scoring")));
  const limited = await createYesNoFilter({}, mockReranq({ prefixLength: 8100 }).request)("q", "d");
  assert.equal(limited.promptTokens, 8191);
  assert.equal(limited.documentTokens, 89);
});

void test("Reranq deduplicates variants, skips multi-token variants, and supports endpoint overrides", async () => {
  const { calls, request } = mockReranq({
    variants: { Yes: [1], " Yes": [1], No: [3], " No": [4, 5] },
  });
  const result = await createYesNoFilter(
    { PI_SEARCH_FILTER_BASE_URL: "http://custom:8151/v1/", PI_SEARCH_FILTER_MODEL: "custom-model" },
    request,
  )("q", "d");
  assert.equal(result.scoringRequests, 1);
  const scoring = calls.find((c) => c.url.endsWith("/generative_scoring"))!;
  assert.equal(scoring.url, "http://custom:8151/generative_scoring");
  assert.equal(scoring.body.model, "custom-model");
  assert.deepEqual(scoring.body.label_token_ids, [1, 3]);
  await assert.rejects(
    createYesNoFilter(
      {},
      mockReranq({ variants: { Yes: [1, 2], " Yes": [2, 3], No: [3], " No": [4] } }).request,
    )("q", "d"),
    /single-token Yes/,
  );
});

void test("filter errors propagate rather than returning unfiltered results", async () => {
  const filter = createYesNoFilter(
    {},
    (async () => new Response("unavailable", { status: 503 })) as typeof fetch,
  );
  await assert.rejects(filter("q", "d"), /HTTP 503/);
});

void test("schema and prompts are gated to enabled two-tool mode", () => {
  const old = process.env.PI_SEARCH_FILTER_ENABLED;
  try {
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
    if (old === undefined) delete process.env.PI_SEARCH_FILTER_ENABLED;
    else process.env.PI_SEARCH_FILTER_ENABLED = old;
  }
});

void test("disabled filter and three-tool mode ignore invalid backend settings", () => {
  const oldEnabled = process.env.PI_SEARCH_FILTER_ENABLED;
  const oldBackend = process.env.PI_SEARCH_FILTER_BACKEND;
  try {
    process.env.PI_SEARCH_FILTER_BACKEND = "unknown-backend";
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
      /Unknown PI_SEARCH_FILTER_BACKEND/,
    );
  } finally {
    if (oldEnabled === undefined) delete process.env.PI_SEARCH_FILTER_ENABLED;
    else process.env.PI_SEARCH_FILTER_ENABLED = oldEnabled;
    if (oldBackend === undefined) delete process.env.PI_SEARCH_FILTER_BACKEND;
    else process.env.PI_SEARCH_FILTER_BACKEND = oldBackend;
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
      undefined, { cwd: "." },
      { ...deps,
        generateFilterQuestion: async (query) => { generated++; assert.equal(query, "shared"); return "Generated question?"; },
        documentFilter: async (question) => { scored++; assert.equal(question, "Generated question?"); return filterScoreFromYesProbability(0.9); },
      },
    );
    assert.equal(generated, 1);
    assert.equal(scored, 3);
    assert.equal(generatedResult.details.filter?.inputSource, "qorrect");
    assert.equal(generatedResult.details.filter?.question, "Generated question?");
    await assert.rejects(executeDirectSearchTool(
      params, undefined, { cwd: "." },
      { ...deps, generateFilterQuestion: async () => { throw new Error("generation failed"); } },
    ), /generation failed/);
    const relevance = await executeDirectSearchTool(
      { reason: "test", query: "shared", hits: 3 },
      undefined,
      { cwd: "." },
      { ...deps, questionSource: "general", documentFilter: async (query) => {
        assert.equal(query, "shared");
        return filterScoreFromYesProbability(0.9);
      } },
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
