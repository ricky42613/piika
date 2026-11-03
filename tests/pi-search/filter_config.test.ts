import assert from "node:assert/strict";
import test from "node:test";
import {
  filterPredictor,
  filterQuestionSource,
  filterThreshold,
} from "../../src/pi-search/filter_config";
import {
  resolveFilterBackend,
  QWEN3_FILTER_INSTRUCTION,
  QWEN3_AFFIRMATIVE_INSTRUCTION,
} from "../../src/pi-search/filter_backends";
import {
  createYesNoFilter,
  filterScoreFromYesProbability,
} from "../../src/pi-search/yes_no_filter";
import { registerPiSearchExtension } from "../../src/pi-search/extension";
import { buildMockExtensionConfig } from "../../src/pi-search/config";
import { formatPiSearchPrompt } from "../../src/pi-search/agent_prompt";

void test("all three Qwen question sources use the intended prompt and schema", () => {
  const old = { ...process.env };
  const keys = [
    "PI_SEARCH_FILTER_ENABLED",
    "PI_SEARCH_FILTER_QUESTION_SOURCE",
    "PI_SEARCH_FILTER_PREDICTOR",
    "PI_SEARCH_FILTER_PREDICTOR_BASE_URL",
    "PI_SEARCH_ORIGINAL_QUESTION",
  ];
  try {
    for (const source of ["agent", "qorrect", "general"] as const) {
        const env = {
          PI_SEARCH_FILTER_ENABLED: "1",
          PI_SEARCH_FILTER_QUESTION_SOURCE: source,
          PI_SEARCH_FILTER_PREDICTOR: "qwen",
          PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "http://predictor",
          PI_SEARCH_ORIGINAL_QUESTION: "original",
        };
        const profile = resolveFilterBackend(env);
        assert.equal(filterQuestionSource(env), source);
        assert.equal(filterPredictor(env), "qwen");
        assert.equal(
          profile.instruction,
          source === "general" ? QWEN3_FILTER_INSTRUCTION : QWEN3_AFFIRMATIVE_INSTRUCTION,
        );
        Object.assign(process.env, env);
        const tools: Array<{
          parameters: { properties: Record<string, unknown> };
          promptGuidelines?: string[];
        }> = [];
        registerPiSearchExtension(
          { on: () => {}, registerTool: (t: (typeof tools)[number]) => tools.push(t) } as never,
          {
            toolInterface: "pyserini-rest-2tool",
            resolveConfig: () => buildMockExtensionConfig({ documents: [] }),
          },
        );
        assert.equal("filter_question" in tools[0].parameters.properties, source === "agent");
        assert.equal(
          tools[0].promptGuidelines!.join(" ").includes("filter_question"),
          source === "agent",
        );
        assert.equal(
          formatPiSearchPrompt("q", "plain_minimal").includes("filter_question"),
          source === "agent",
        );
    }
  } finally {
    for (const key of keys) {
      if (old[key] === undefined) delete process.env[key];
      else process.env[key] = old[key];
    }
  }
});

void test("threshold validation and strict boundaries", () => {
  assert.equal(filterThreshold({}), 0.5);
  for (const value of ["NaN", "Infinity", "-0.1", "1.1", "abc"])
    assert.throws(() => filterThreshold({ PI_SEARCH_FILTER_THRESHOLD: value }));
  for (const t of [0, 0.2, 0.5, 1]) {
    assert.equal(filterThreshold({ PI_SEARCH_FILTER_THRESHOLD: String(t) }), t);
    assert.equal(filterScoreFromYesProbability(t, t).keep, false);
  }
  assert.equal(filterScoreFromYesProbability(0.3, 0.2).keep, true);
  assert.equal(filterScoreFromYesProbability(0.7, 0.8).keep, false);
});

void test("predictor defaults to Qwen, rejects removed backends, and source is independent", () => {
  assert.equal(filterPredictor({}), "qwen");
  assert.equal(filterPredictor({ PI_SEARCH_FILTER_PREDICTOR: "jev" }), "jev");
  assert.equal(filterPredictor({ PI_SEARCH_FILTER_PREDICTOR: "pplx_decider" }), "pplx_decider");
  assert.equal(filterQuestionSource({ PI_SEARCH_FILTER_PREDICTOR: "qwen" }), "agent");
  assert.throws(() => filterPredictor({ PI_SEARCH_FILTER_PREDICTOR: "reranq" }), /must be qwen, jev or pplx_decider/);
  assert.throws(() => filterPredictor({ PI_SEARCH_FILTER_PREDICTOR: "qorrect" }), /question source only/);
  assert.throws(() => filterPredictor({ PI_SEARCH_FILTER_BACKEND: "reranq" }), /no longer supported/);
  assert.throws(() => filterPredictor({ PI_SEARCH_FILTER_BACKEND: "qwen3-reranker" }), /no longer supported/);
  assert.throws(() => filterPredictor({ PI_SEARCH_FILTER_PREDICTOR: "bad" }));
});

void test("score transport uses source-specific instruction and configured threshold", async () => {
  for (const source of ["agent", "qorrect", "general"]) {
    const env = {
      PI_SEARCH_FILTER_PREDICTOR: "qwen",
      PI_SEARCH_FILTER_QUESTION_SOURCE: source,
      PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "http://score",
      PI_SEARCH_FILTER_PREDICTOR_API: "score",
      PI_SEARCH_FILTER_THRESHOLD: "0.8",
    };
    const filter = createYesNoFilter(env, (async (url, init) => {
      assert.ok(typeof url === "string");
      assert.ok(init);
      const bodyRaw = init.body;
      assert.equal(typeof bodyRaw, "string");
      const body = JSON.parse(bodyRaw as string);
      if (url.endsWith("/tokenize")) return Response.json({ tokens: [1, 2] });
      assert.equal(
        body.instruction,
        source === "general" ? QWEN3_FILTER_INSTRUCTION : QWEN3_AFFIRMATIVE_INSTRUCTION,
      );
      return Response.json({ data: [{ index: 0, score: 0.7 }] });
    }) as typeof fetch);
    const result = await filter("query", "document");
    assert.equal(result.keep, false);
    assert.equal(result.threshold, 0.8);
    assert.equal(result.questionSource, source);
    assert.equal(result.predictor, "qwen");
  }
});

void test("disabled mode ignores all invalid filter settings and requires no services", () => {
  const values = {
    PI_SEARCH_FILTER_ENABLED: "0",
    PI_SEARCH_FILTER_PREDICTOR: "bad",
    PI_SEARCH_FILTER_QUESTION_SOURCE: "bad",
    PI_SEARCH_FILTER_THRESHOLD: "bad",
    PI_SEARCH_FILTER_GENERATOR_BASE_URL: "bad",
  };
  const old = { ...process.env };
  try {
    Object.assign(process.env, values);
    const tools: Array<{ parameters: { properties: Record<string, unknown> } }> = [];
    registerPiSearchExtension(
      { on: () => {}, registerTool: (t: (typeof tools)[number]) => tools.push(t) } as never,
      { resolveConfig: () => buildMockExtensionConfig({ documents: [] }) },
    );
    assert.equal("filter_question" in tools[0].parameters.properties, false);
    assert.doesNotMatch(formatPiSearchPrompt("q", "plain_minimal"), /filter_question/);
  } finally {
    for (const key of Object.keys(values)) {
      if (old[key] === undefined) delete process.env[key];
      else process.env[key] = old[key];
    }
  }
});
