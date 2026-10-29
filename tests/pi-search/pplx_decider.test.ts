import assert from "node:assert/strict";
import test from "node:test";

import { createPplxDeciderScorer } from "../../src/pi-search/reranking/pplx_decider";

const config = { baseUrl: "http://127.0.0.1:8002/v1/", model: "pplx-decider-flash-v1.2" };
const criterion = { question: "Does this document identify\nthe engine designer?" };
const candidates = Object.freeze([
  Object.freeze({ docid: "a", title: "A", text: "First document", retrievalScore: 10 }),
  Object.freeze({ docid: "b", text: "Second\ndocument", retrievalScore: 8 }),
]);

function result(noul: unknown, model = config.model) {
  return Response.json({
    model,
    answers: { relevance: { type: "noul", noul } },
    usage: { input_tokens: 123, output_tokens: 0 },
  });
}

void test("Decider scores documents separately and preserves all candidate fields without mutation", async () => {
  let calls = 0;
  const scorer = createPplxDeciderScorer(config, async (url, init) => {
    const index = calls++;
    assert.equal(String(url), "http://127.0.0.1:8002/v1/decisions");
    assert.equal(init?.method, "POST");
    assert.deepEqual(JSON.parse(String(init?.body)), {
      model: config.model,
      state: candidates[index].text,
      questions: { relevance: { type: "noul", instructions: criterion.question } },
    });
    return result(index === 0 ? 0.1 : 0.9);
  });
  const scored = await scorer.score(criterion, candidates);
  assert.deepEqual(scored, [
    { ...candidates[0], rerankScore: 0.1 },
    { ...candidates[1], rerankScore: 0.9 },
  ]);
  assert.equal(calls, 2);
  assert.notStrictEqual(scored[0], candidates[0]);
  assert.ok(candidates.every((candidate) => !("rerankScore" in candidate)));
});

void test("Decider bounds concurrency and preserves order when responses complete out of order", async () => {
  const inputs = Array.from({ length: 6 }, (_, index) => ({
    docid: String(index),
    text: String(index),
  }));
  const pending = new Map<number, (response: Response) => void>();
  let active = 0;
  let peak = 0;
  const scorer = createPplxDeciderScorer({ ...config, concurrency: 2 }, async (_url, init) => {
    const index = Number(JSON.parse(String(init?.body)).state);
    active += 1;
    peak = Math.max(peak, active);
    const response = await new Promise<Response>((resolve) => pending.set(index, resolve));
    active -= 1;
    return response;
  });
  const scoring = scorer.score(criterion, inputs);
  const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(pending.size, 2);
  // Keep the first request pending while the other worker progresses.
  for (let index = 1; index < inputs.length; index += 1) {
    assert.ok(pending.has(index));
    pending.get(index)!(result(index / 10));
    await tick();
  }
  pending.get(0)!(result(0));
  const scored = await scoring;
  assert.equal(peak, 2);
  assert.deepEqual(
    scored.map((candidate) => candidate.docid),
    inputs.map((candidate) => candidate.docid),
  );
  assert.deepEqual(
    scored.map((candidate) => candidate.rerankScore),
    [0, 0.1, 0.2, 0.3, 0.4, 0.5],
  );
});

void test("Decider aborts in-flight work after a failure and does not start queued candidates", async () => {
  let calls = 0;
  let aborted = false;
  const scorer = createPplxDeciderScorer({ ...config, concurrency: 2 }, async (_url, init) => {
    calls += 1;
    if (calls === 1) return new Response("failure", { status: 500 });
    const signal = init?.signal;
    assert.ok(signal);
    return new Promise<Response>((_resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => {
          aborted = true;
          reject(signal.reason);
        },
        { once: true },
      );
    });
  });
  await assert.rejects(scorer.score(criterion, [...candidates, ...candidates]), /HTTP 500/);
  assert.equal(calls, 2);
  assert.equal(aborted, true);
});

void test("concurrency one retains sequential request behavior", async () => {
  let calls = 0;
  const scorer = createPplxDeciderScorer({ ...config, concurrency: 1 }, async () => {
    calls += 1;
    return new Response("failure", { status: 500 });
  });
  await assert.rejects(scorer.score(criterion, candidates), /HTTP 500/);
  assert.equal(calls, 1);
});

void test("empty candidates require no HTTP requests", async () => {
  const scorer = createPplxDeciderScorer(config, async () => {
    throw new Error("must not call HTTP");
  });
  assert.deepEqual(await scorer.score(criterion, []), []);
});

void test("Decider accepts probability endpoints zero and one", async () => {
  for (const probability of [0, 1]) {
    const scorer = createPplxDeciderScorer(config, async () => result(probability));
    assert.equal(
      (await scorer.score(criterion, candidates.slice(0, 1)))[0].rerankScore,
      probability,
    );
  }
});

void test("Decider rejects malformed probabilities, wrong models and invalid JSON", async () => {
  const responses = [
    result(-0.1),
    result(1.1),
    result("0.9"),
    result(null),
    result(0.5, "wrong-model"),
    Response.json({ model: config.model, answers: {} }),
    Response.json({ model: config.model, answers: { relevance: { type: "score", noul: 0.5 } } }),
    new Response("not json"),
  ];
  for (const response of responses) {
    const scorer = createPplxDeciderScorer(config, async () => response);
    await assert.rejects(scorer.score(criterion, candidates));
  }
});

void test("HTTP failure stops scoring without returning partial results or exposing response body", async () => {
  let calls = 0;
  const scorer = createPplxDeciderScorer(config, async () => {
    calls += 1;
    return calls === 1 ? result(0.5) : new Response("private details", { status: 500 });
  });
  await assert.rejects(scorer.score(criterion, candidates), {
    message: "Decider scoring failed: HTTP 500.",
  });
  assert.equal(calls, 2);
});

void test("Decider propagates network errors", async () => {
  const error = new Error("connection refused");
  const scorer = createPplxDeciderScorer(config, async () => {
    throw error;
  });
  await assert.rejects(scorer.score(criterion, candidates), (actual) => actual === error);
});

void test("Decider forwards cancellation and stops before subsequent candidates", async () => {
  const controller = new AbortController();
  let calls = 0;
  const scorer = createPplxDeciderScorer(config, async (_url, init) => {
    calls += 1;
    controller.abort();
    assert.equal(init?.signal?.aborted, true);
    return result(0.5);
  });
  await assert.rejects(scorer.score(criterion, candidates, controller.signal), {
    name: "AbortError",
  });
  assert.equal(calls, 1);
});

void test("Decider rejects an already aborted request before calling HTTP", async () => {
  const scorer = createPplxDeciderScorer(config, async () => {
    throw new Error("must not call HTTP");
  });
  await assert.rejects(scorer.score(criterion, candidates, AbortSignal.abort()), {
    name: "AbortError",
  });
});

void test("Decider times out stalled requests", async () => {
  const keepAlive = setTimeout(() => {}, 1000);
  try {
    const scorer = createPplxDeciderScorer({ ...config, timeoutMs: 10 }, async (_url, init) => {
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

void test("Decider validates configuration and empty questions", async () => {
  for (const override of [
    { baseUrl: "file:///tmp/model" },
    { model: " " },
    { timeoutMs: 0 },
    { concurrency: 0 },
    { concurrency: 1.5 },
  ]) {
    assert.throws(() => createPplxDeciderScorer({ ...config, ...override }));
  }
  const scorer = createPplxDeciderScorer(config, async () => {
    throw new Error("must not call HTTP");
  });
  await assert.rejects(
    scorer.score({ question: " \n " }, candidates),
    /question must be non-empty/,
  );
});
