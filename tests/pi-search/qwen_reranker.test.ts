import assert from "node:assert/strict";
import test from "node:test";

import { createQwenRerankerScorer } from "../../src/pi-search/reranking/qwen_reranker";

const config = { baseUrl: "http://127.0.0.1:8003/", model: "qwen3-reranker" };
const criterion = { question: "Does this document identify\nthe engine designer?" };
const candidates = Object.freeze([
  Object.freeze({ docid: "a", title: "A", text: "First document", retrievalScore: 10 }),
  Object.freeze({ docid: "b", text: "Second\ndocument", retrievalScore: 8 }),
]);

function result(data: unknown, model = config.model) {
  return Response.json({ model, data });
}

void test("Qwen batches all documents and maps out-of-order scores without mutating inputs", async () => {
  let calls = 0;
  const scorer = createQwenRerankerScorer(config, async (url, init) => {
    calls += 1;
    assert.equal(String(url), "http://127.0.0.1:8003/score");
    assert.equal(init?.method, "POST");
    assert.deepEqual(JSON.parse(String(init?.body)), {
      model: config.model,
      queries: criterion.question,
      documents: ["A\nFirst document", "Second\ndocument"],
      instruction: "Determine whether the document satisfies the requirement in the query.",
      use_activation: true,
    });
    return result([
      { index: 1, score: 0.9 },
      { index: 0, score: 0.1 },
    ]);
  });
  const scored = await scorer.score(criterion, candidates);
  assert.deepEqual(scored, [
    { ...candidates[0], rerankScore: 0.1 },
    { ...candidates[1], rerankScore: 0.9 },
  ]);
  assert.equal(calls, 1);
  assert.notStrictEqual(scored[0], candidates[0]);
  assert.ok(candidates.every((candidate) => !("rerankScore" in candidate)));
});

void test("Qwen skips HTTP for empty candidates", async () => {
  const scorer = createQwenRerankerScorer(config, async () => {
    throw new Error("unexpected HTTP");
  });
  assert.deepEqual(await scorer.score(criterion, []), []);
});

void test("Qwen rejects missing, duplicate, invalid indices and invalid scores", async () => {
  for (const data of [
    [],
    null,
    [{ index: 0, score: 0.5 }],
    [
      { index: 0, score: 0.5 },
      { index: 0, score: 0.6 },
    ],
    [
      { index: 0, score: 0.5 },
      { index: 2, score: 0.6 },
    ],
    [
      { index: -1, score: 0.5 },
      { index: 1, score: 0.6 },
    ],
    [
      { index: 0.5, score: 0.5 },
      { index: 1, score: 0.6 },
    ],
    [
      { index: 0, score: "0.5" },
      { index: 1, score: 0.6 },
    ],
    [
      { index: 0, score: null },
      { index: 1, score: 0.6 },
    ],
  ]) {
    const scorer = createQwenRerankerScorer(config, async () => result(data));
    await assert.rejects(scorer.score(criterion, candidates), /Invalid Qwen reranker response/);
  }
});

void test("Qwen rejects wrong model and malformed JSON", async () => {
  for (const response of [result([], "wrong-model"), new Response("not JSON")]) {
    const scorer = createQwenRerankerScorer(config, async () => response);
    await assert.rejects(scorer.score(criterion, candidates));
  }
});

void test("Qwen reports HTTP failures without leaking response body", async () => {
  const scorer = createQwenRerankerScorer(
    config,
    async () => new Response("private details", { status: 500 }),
  );
  await assert.rejects(scorer.score(criterion, candidates), {
    message: "Qwen reranker scoring failed: HTTP 500.",
  });
});

void test("Qwen propagates network failures", async () => {
  const error = new Error("connection refused");
  const scorer = createQwenRerankerScorer(config, async () => {
    throw error;
  });
  await assert.rejects(scorer.score(criterion, candidates), (actual) => actual === error);
});

void test("Qwen forwards cancellation and rejects already aborted calls", async () => {
  const controller = new AbortController();
  let calls = 0;
  const scorer = createQwenRerankerScorer(config, async (_url, init) => {
    calls += 1;
    controller.abort();
    assert.equal(init?.signal?.aborted, true);
    init?.signal?.throwIfAborted();
    throw new Error("unreachable");
  });
  await assert.rejects(scorer.score(criterion, candidates, controller.signal), {
    name: "AbortError",
  });
  await assert.rejects(scorer.score(criterion, candidates, controller.signal), {
    name: "AbortError",
  });
  assert.equal(calls, 1);
});

void test("Qwen times out stalled requests", async () => {
  const keepAlive = setTimeout(() => {}, 1000);
  try {
    const scorer = createQwenRerankerScorer({ ...config, timeoutMs: 10 }, async (_url, init) => {
      const signal = init?.signal;
      assert.ok(signal);
      return new Promise<Response>((_resolve, reject) => {
        if (signal.aborted) reject(signal.reason);
        else signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    });
    await assert.rejects(scorer.score(criterion, candidates), { name: "TimeoutError" });
  } finally {
    clearTimeout(keepAlive);
  }
});

void test("Qwen validates configuration and empty criterion", async () => {
  for (const override of [{ baseUrl: "file:///tmp/model" }, { model: " " }, { timeoutMs: 0 }]) {
    assert.throws(() => createQwenRerankerScorer({ ...config, ...override }));
  }
  const scorer = createQwenRerankerScorer(config, async () => {
    throw new Error("unexpected HTTP");
  });
  await assert.rejects(scorer.score({ question: " " }, candidates), /question must be non-empty/);
});
