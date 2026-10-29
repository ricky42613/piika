import assert from "node:assert/strict";
import test from "node:test";

import { createSearchReranker } from "../../src/pi-search/reranking/factory";
import { buildCustomizedCriterion } from "../../src/pi-search/reranking/criterion";

const pplx = {
  provider: "pplx-decider" as const,
  baseUrl: "http://127.0.0.1:8002/v1",
  model: "pplx-decider-flash-v1.2",
};
const qwen = {
  provider: "qwen-reranker" as const,
  baseUrl: "http://127.0.0.1:8003",
  model: "qwen3-reranker",
};
const customized = {
  mode: "customized" as const,
  candidateK: 50,
  scorer: pplx,
  generator: {
    provider: "qorrect" as const,
    baseUrl: "http://127.0.0.1:8001/v1",
    model: "qorrect",
  },
};

void test("disabled reranking creates no reranker on either tool interface", () => {
  for (const toolInterface of ["pyserini-rest-2tool", "pi-serini-3tool"] as const) {
    assert.equal(createSearchReranker(undefined, toolInterface), undefined);
    assert.equal(createSearchReranker({ mode: "off" }, toolInterface), undefined);
  }
});

void test("enabled reranking refuses the 3-tool interface", () => {
  for (const config of [
    customized,
    { mode: "fixed-template" as const, candidateK: 50, scorer: qwen },
  ]) {
    assert.throws(() => createSearchReranker(config, "pi-serini-3tool"), /only.*2tool/);
  }
});

void test("factory selects the configured scorer and reuses its client", async () => {
  for (const scorer of [pplx, qwen]) {
    let calls = 0;
    const runtime = createSearchReranker(
      { mode: "fixed-template", candidateK: 50, scorer },
      "pyserini-rest-2tool",
      {
        fetchImpl: async (url) => {
          calls += 1;
          assert.equal(
            String(url),
            `${scorer.baseUrl}/${scorer.provider === "pplx-decider" ? "decisions" : "score"}`,
          );
          return Response.json(
            scorer.provider === "pplx-decider"
              ? { model: scorer.model, answers: { relevance: { type: "noul", noul: 0.8 } } }
              : { model: scorer.model, data: [{ index: 0, score: 0.8 }] },
          );
        },
      },
    );
    assert.ok(runtime);
    assert.equal(calls, 0);
    const first = runtime.buildSearchOptions();
    const second = runtime.buildSearchOptions();
    assert.equal(first.mode, "fixed-template");
    assert.equal(first.candidateK, 50);
    assert.strictEqual(first.scorer, second.scorer);
    assert.deepEqual(
      await first.scorer.score({ question: "Relevant?" }, [{ docid: "a", text: "A" }]),
      [{ docid: "a", text: "A", rerankScore: 0.8 }],
    );
  }
});

void test("customized reranker reuses clients without retaining earlier search context", () => {
  const runtime = createSearchReranker(customized, "pyserini-rest-2tool");
  assert.ok(runtime);
  const first = runtime.buildSearchOptions({
    originalQuestion: "First task",
    searchReason: "Explore",
  });
  const second = runtime.buildSearchOptions({
    originalQuestion: "Second task",
    searchReason: "Verify",
  });
  assert.equal(first.mode, "customized");
  assert.equal(second.mode, "customized");
  if (first.mode !== "customized" || second.mode !== "customized") throw new Error("wrong mode");
  assert.strictEqual(first.scorer, second.scorer);
  assert.strictEqual(first.generator, second.generator);
  assert.equal(first.originalQuestion, "First task");
  assert.equal(second.originalQuestion, "Second task");
  assert.equal(second.searchReason, "Verify");
});

void test("customized reranker rejects missing original question", () => {
  const runtime = createSearchReranker(customized, "pyserini-rest-2tool");
  assert.ok(runtime);
  assert.throws(() => runtime.buildSearchOptions(), /requires the original question/);
  assert.throws(
    () => runtime.buildSearchOptions({ originalQuestion: " " }),
    /requires the original question/,
  );
});

void test("Qorrect credentials are resolved from the named environment variable", async () => {
  const config = {
    ...customized,
    generator: { ...customized.generator, tokenEnv: "QORRECT_KEY", temperature: 0 },
  };
  const runtime = createSearchReranker(config, "pyserini-rest-2tool", {
    env: { QORRECT_KEY: "test-key" },
    fetchImpl: async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8001/v1/chat/completions");
      assert.equal(new Headers(init?.headers).get("authorization"), "Bearer test-key");
      assert.equal(JSON.parse(String(init?.body)).temperature, 0);
      return Response.json({
        choices: [{ message: { content: "Relevant to the engine?" }, finish_reason: "stop" }],
      });
    },
  });
  const options = runtime!.buildSearchOptions({ originalQuestion: "Who designed it?" });
  if (options.mode !== "customized") throw new Error("wrong mode");
  assert.deepEqual(
    await buildCustomizedCriterion(
      { currentQuery: "engine", originalQuestion: options.originalQuestion },
      options.generator,
    ),
    { question: "Relevant to the engine?" },
  );
});

void test("missing configured credentials fail at initialization", () => {
  assert.throws(
    () =>
      createSearchReranker(
        {
          ...customized,
          generator: { ...customized.generator, tokenEnv: "QORRECT_KEY" },
        },
        "pyserini-rest-2tool",
        { env: {} },
      ),
    /Missing Qorrect API key.*QORRECT_KEY/,
  );
});
