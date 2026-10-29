import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMockExtensionConfig,
  parsePiSearchExtensionConfig,
  parsePiSearchRerankingConfig,
} from "../../src/pi-search/config";

const backend = { kind: "mock", documents: [] };
const pplx = {
  provider: "pplx-decider",
  baseUrl: "http://127.0.0.1:8002/v1",
  model: "pplx-decider-flash-v1.2",
};
const qwen = {
  provider: "qwen-reranker",
  baseUrl: "http://127.0.0.1:8003",
  model: "qwen3-reranker",
};
const generator = {
  provider: "qorrect",
  baseUrl: "http://127.0.0.1:8001/v1",
  model: "qorrect",
};
const customized = { mode: "customized", candidateK: 50, generator, scorer: pplx };

function parse(reranking: unknown) {
  return parsePiSearchExtensionConfig(JSON.stringify({ backend, reranking }));
}

void test("standalone reranking JSON accepts the same modes as extension config", () => {
  for (const config of [
    { mode: "off" },
    { mode: "fixed-template", candidateK: 50, scorer: qwen },
    customized,
  ]) {
    assert.deepEqual(parsePiSearchRerankingConfig(JSON.stringify(config)), config);
  }
});

void test("standalone reranking JSON rejects invalid input without echoing secrets", () => {
  for (const text of [
    '{"apiKey":"top-secret",',
    JSON.stringify({ ...customized, generator: { ...generator, apiKey: "top-secret" } }),
    JSON.stringify({ ...customized, scorer: qwen }),
    JSON.stringify({ backend, reranking: customized }),
  ]) {
    assert.throws(
      () => parsePiSearchRerankingConfig(text),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.doesNotMatch(error.message, /top-secret/);
        return true;
      },
    );
  }
});

void test("existing configs remain unchanged when reranking is omitted", () => {
  const config = buildMockExtensionConfig({ documents: [] });
  assert.deepEqual(parsePiSearchExtensionConfig(JSON.stringify(config)), config);
  assert.equal(config.reranking, undefined);
});

void test("reranking can be explicitly disabled", () => {
  assert.deepEqual(parse({ mode: "off" }).reranking, { mode: "off" });
});

void test("fixed-template accepts either scorer without a generator", () => {
  for (const scorer of [pplx, qwen]) {
    const reranking = { mode: "fixed-template", candidateK: 50, scorer };
    assert.deepEqual(parse(reranking).reranking, reranking);
  }
});

void test("customized accepts Qorrect plus PPLX and optional client settings", () => {
  const reranking = {
    ...customized,
    generator: {
      ...generator,
      temperature: 0,
      maxTokens: 64,
      timeoutMs: 30000,
      tokenEnv: "QORRECT_API_KEY",
    },
    scorer: { ...pplx, concurrency: 4, timeoutMs: 60000 },
  };
  assert.deepEqual(parse(reranking).reranking, reranking);
});

const invalidConfigs: Array<[string, unknown]> = [
  ["unknown mode", { mode: "automatic" }],
  ["missing candidate depth", { mode: "fixed-template", scorer: pplx }],
  ["zero candidate depth", { ...customized, candidateK: 0 }],
  ["fractional candidate depth", { ...customized, candidateK: 1.5 }],
  ["missing scorer", { mode: "fixed-template", candidateK: 50 }],
  ["missing generator", { mode: "customized", candidateK: 50, scorer: pplx }],
  ["Qwen scorer in customized mode", { ...customized, scorer: qwen }],
  ["generator in fixed mode", { ...customized, mode: "fixed-template" }],
  ["settings in off mode", { mode: "off", candidateK: 50 }],
  ["non-HTTP endpoint", { ...customized, scorer: { ...pplx, baseUrl: "file:///tmp/model" } }],
  ["blank model name", { ...customized, scorer: { ...pplx, model: " " } }],
  ["zero concurrency", { ...customized, scorer: { ...pplx, concurrency: 0 } }],
  ["zero timeout", { ...customized, scorer: { ...pplx, timeoutMs: 0 } }],
  ["invalid temperature", { ...customized, generator: { ...generator, temperature: 3 } }],
  ["zero output tokens", { ...customized, generator: { ...generator, maxTokens: 0 } }],
  ["inline API key", { ...customized, generator: { ...generator, apiKey: "not-a-real-key" } }],
  ["per-search context in config", { ...customized, originalQuestion: "Task" }],
];

for (const [name, config] of invalidConfigs) {
  void test(`rerank config rejects ${name}`, () => {
    assert.throws(() => parse(config), /Invalid PI_SEARCH_EXTENSION_CONFIG/);
  });
}
