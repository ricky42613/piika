import assert from "node:assert/strict";
import test from "node:test";
import { buildMockExtensionConfig } from "../../src/pi-search/config";
import { registerPiSearchExtension } from "../../src/pi-search/extension";
import { PiSearchBackendRuntime } from "../../src/pi-search/searcher/runtime";
import { SearchSessionStore } from "../../src/pi-search/search_cache";
import { ManagedTempSpillDir } from "../../src/pi-search/spill";
import { executeDirectSearchTool } from "../../src/pi-search/tool_handlers";
import { createYesNoFilter, JEV_GENERAL_PROMPT } from "../../src/pi-search/yes_no_filter";

const jevEnv = {
  PI_SEARCH_FILTER_PREDICTOR: "jev",
  PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "https://api.typesafe.ai/v1",
  PI_SEARCH_FILTER_PREDICTOR_MODEL: "jev-latest",
};

void test("JEV sends fixed general question without criteria, keeping true > false", async () => {
  const calls: Array<{ url: string; body: Record<string, unknown>; auth: string | null }> = [];
  const filter = createYesNoFilter(
    { ...jevEnv, PI_SEARCH_FILTER_PREDICTOR_API_KEY: "secret" },
    (async (url, init) => {
      const body = JSON.parse(init!.body as string) as Record<string, unknown>;
      calls.push({ url: String(url), body, auth: new Headers(init?.headers).get("Authorization") });
      assert.equal(url, "https://api.typesafe.ai/v1/systemone");
      assert.equal(body.model, "jev-latest");
      assert.deepEqual(body.state, {
        query: "goldfish growth",
        candidate_passage: "positive passage",
      });
      assert.deepEqual(body.questions, {
        relevant: { type: "noul", instructions: JEV_GENERAL_PROMPT },
      });
      return Response.json({
        model: "jev-1.13.0",
        answers: { relevant: { noul: 0.7 } },
        usage: { input_tokens: 123 },
      });
    }) as typeof fetch,
  );
  const result = await filter("goldfish growth", "positive passage");
  assert.equal(result.keep, true);
  assert.equal(result.yesProbability, 0.7);
  assert.ok(Math.abs(result.noProbability - 0.3) < 1e-12);
  assert.equal(result.predictor, "jev");
  assert.equal(result.questionSource, "general");
  assert.equal(result.scoringApi, "systemone");
  assert.equal(result.promptTokens, 123);
  assert.equal(calls[0].auth, "Bearer secret");
});

void test("JEV schemas use regular search arguments, with optional clue-doc interface", () => {
  const saved = { ...process.env };
  try {
    Object.assign(process.env, {
      PI_SEARCH_FILTER_ENABLED: "1",
      PI_SEARCH_FILTER_PREDICTOR: "jev",
      PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect", // Must not initialize a Qorrect generator.
      PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "https://api.typesafe.ai/v1",
    });
    delete process.env.PI_SEARCH_ORIGINAL_QUESTION;
    for (const toolInterface of ["pyserini-rest-2tool", "pyserini-rest-2tool-clue-docs"] as const) {
      const tools: Array<{
        name: string;
        parameters: { required?: string[]; properties?: Record<string, unknown> };
        promptGuidelines?: string[];
      }> = [];
      registerPiSearchExtension(
        { on: () => {}, registerTool: (tool: (typeof tools)[number]) => tools.push(tool) } as never,
        {
          toolInterface,
          resolveConfig: () => buildMockExtensionConfig({ documents: [] }),
        },
      );
      assert.deepEqual(tools.map((tool) => tool.name), ["search", "read_document"]);
      const fields = Object.keys(tools[0].parameters.properties ?? {});
      assert.deepEqual(fields, toolInterface === "pyserini-rest-2tool" ? ["reason", "query", "hits"] : ["reason", "query", "hits", "clue_docs"]);
      const guidance = tools[0].promptGuidelines?.join(" ") ?? "";
      assert.match(guidance, /Does the document contain information that directly helps answer the query\?/);
      assert.doesNotMatch(guidance, /provide relevance_assess_question|provide filter_question/);
    }
  } finally {
    for (const key of [
      "PI_SEARCH_FILTER_ENABLED",
      "PI_SEARCH_FILTER_PREDICTOR",
      "PI_SEARCH_FILTER_QUESTION_SOURCE",
      "PI_SEARCH_FILTER_PREDICTOR_BASE_URL",
      "PI_SEARCH_ORIGINAL_QUESTION",
    ]) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key];
    }
  }
});

void test("JEV uses search query for scoring and REREANK_DEPTH for candidates", async () => {
  const runtime = new PiSearchBackendRuntime(
    buildMockExtensionConfig({
      documents: [
        { docid: "a", text: "shared jev a" },
        { docid: "b", text: "shared jev b" },
        { docid: "c", text: "shared jev c" },
        { docid: "d", text: "shared jev d" },
      ],
    }),
  );
  const spillDir = new ManagedTempSpillDir("jev-depth-test-");
  let scored = 0;
  const deps = {
    backendRuntime: runtime,
    searchStore: new SearchSessionStore(),
    spillDir,
    nextSpillSequence: () => 1,
    rerankDepth: 3,
    predictor: "jev" as const,
    documentFilter: async (query: string, doc: string) => {
      scored++;
      assert.equal(query, "shared jev");
      return {
        yesProbability: doc.includes("c") ? 0.4 : 0.8,
        noProbability: doc.includes("c") ? 0.6 : 0.2,
        yesLogprob: null,
        noLogprob: null,
        keep: !doc.includes("c"),
        probabilitySemantics: "normalized_yes_no_tokens" as const,
      };
    },
  };
  try {
    const result = await executeDirectSearchTool(
      { reason: "test", query: "shared jev", hits: 2 },
      undefined,
      { cwd: "." },
      deps,
    );
    assert.equal(scored, 3);
    assert.equal(result.details.filter?.question, JEV_GENERAL_PROMPT);
    assert.equal(result.details.filter?.inputSource, "jev");
    assert.equal(result.details.filter?.candidateCount, 3);
    assert.equal(result.details.filter?.rereankDepth, 3);
    assert.equal(result.details.filter?.returnedCount, 2);
    assert.deepEqual(result.details.retrievedDocids, ["a", "b"]);
    await assert.rejects(
      executeDirectSearchTool({ reason: "test", query: " " }, undefined, { cwd: "." }, deps),
      /query must be a non-empty string/,
    );
  } finally {
    runtime.dispose();
    spillDir.cleanup();
  }
});
