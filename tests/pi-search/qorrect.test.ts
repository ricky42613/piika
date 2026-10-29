import assert from "node:assert/strict";
import test from "node:test";

import { buildCustomizedCriterion } from "../../src/pi-search/reranking/criterion";
import {
  buildQorrectPrompt,
  createQorrectGenerator,
} from "../../src/pi-search/reranking/qorrect";

const context = {
  originalQuestion: "Who designed the analytical engine?",
  currentQuery: "analytical engine designer",
};
const config = { baseUrl: "http://localhost:8000/v1/", model: "qorrect" };

function completion(content: unknown, finishReason = "stop") {
  return Response.json({ choices: [{ message: { content }, finish_reason: finishReason }] });
}

void test("Qorrect client sends the training prompt with thinking disabled and preserves full output", async () => {
  const content = "Does this document identify the designer\nof the analytical engine?";
  let calls = 0;
  const generator = createQorrectGenerator(config, async (url, init) => {
    calls += 1;
    assert.equal(String(url), "http://localhost:8000/v1/chat/completions");
    assert.equal(init?.method, "POST");
    assert.deepEqual(init?.headers, { "content-type": "application/json" });
    assert.deepEqual(JSON.parse(String(init?.body)), {
      model: "qorrect",
      messages: [{ role: "user", content: buildQorrectPrompt(context) }],
      chat_template_kwargs: { enable_thinking: false },
      max_tokens: 64,
      temperature: 1,
      stream: false,
      n: 1,
    });
    return completion(content);
  });
  assert.deepEqual(await buildCustomizedCriterion(context, generator), { question: content });
  assert.equal(calls, 1);
});

void test("Qorrect client supports explicit generation settings and authentication", async () => {
  const generator = createQorrectGenerator({
    ...config, maxTokens: 128, temperature: 0, apiKey: "test-key",
  }, async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    assert.equal(body.max_tokens, 128);
    assert.equal(body.temperature, 0);
    assert.equal(new Headers(init?.headers).get("authorization"), "Bearer test-key");
    return completion("A question?");
  });
  assert.equal(await generator.generateQuestion(context), "A question?");
});

void test("Qorrect client reports HTTP errors without including response bodies", async () => {
  const generator = createQorrectGenerator(config, async () =>
    new Response("private server details", { status: 401 }));
  await assert.rejects(generator.generateQuestion(context), {
    message: "Qorrect generation failed: HTTP 401.",
  });
});

void test("Qorrect client rejects malformed, empty, or truncated completions", async () => {
  const responses = [
    Response.json({}),
    Response.json({ choices: [] }),
    Response.json({ choices: [null] }),
    completion("Partial question", "length"),
    completion(null),
    completion(" \n "),
    new Response("not json"),
  ];
  for (const response of responses) {
    const generator = createQorrectGenerator(config, async () => response);
    await assert.rejects(generator.generateQuestion(context));
  }
});

void test("Qorrect client propagates network failures", async () => {
  const error = new Error("connection refused");
  const generator = createQorrectGenerator(config, async () => { throw error; });
  await assert.rejects(generator.generateQuestion(context), (actual) => actual === error);
});

void test("Qorrect client forwards cancellation to HTTP requests", async () => {
  const controller = new AbortController();
  const generator = createQorrectGenerator(config, async (_url, init) => {
    const signal = init?.signal;
    assert.ok(signal);
    controller.abort();
    assert.equal(signal.aborted, true);
    signal.throwIfAborted();
    throw new Error("unreachable");
  });
  await assert.rejects(generator.generateQuestion(context, controller.signal), { name: "AbortError" });
});

void test("Qorrect client times out stalled HTTP requests", async () => {
  // Keep the event loop alive while AbortSignal.timeout's unref'ed timer runs.
  const keepAlive = setTimeout(() => {}, 1000);
  try {
    const generator = createQorrectGenerator({ ...config, timeoutMs: 10 }, async (_url, init) => {
      const signal = init?.signal;
      assert.ok(signal);
      return new Promise<Response>((_resolve, reject) => {
        if (signal.aborted) reject(signal.reason);
        else signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    });
    await assert.rejects(generator.generateQuestion(context), { name: "TimeoutError" });
  } finally {
    clearTimeout(keepAlive);
  }
});

void test("Qorrect client validates configuration before making requests", () => {
  for (const override of [
    { baseUrl: "file:///tmp/model" },
    { model: " " },
    { maxTokens: 0 },
    { maxTokens: 1.5 },
    { temperature: NaN },
    { temperature: 3 },
    { timeoutMs: 0 },
  ]) {
    assert.throws(() => createQorrectGenerator({ ...config, ...override }));
  }
});

void test("Qorrect prompt matches the saved training prompt without seen documents", () => {
  const prompt = buildQorrectPrompt({
    originalQuestion: "Who designed the analytical engine?",
    currentQuery: "analytical engine designer",
    searchReason: "Find evidence identifying the designer",
  });

  assert.equal(prompt, `You are an agent performing multi-step search.

Original question:
Who designed the analytical engine?

Reason for current search:
Find evidence identifying the designer
Current search query:
analytical engine designer

Write ONE complete yes/no question to filter individual retrieved documents.

Choose the scope of the question based on the CURRENT search state and information need. The question may be broad when exploring possible leads or narrower when checking a particular fact, candidate, or relationship.

Rules:
- Output only ONE complete yes/no question, with no answer, explanation, or labels.
- The question must be answerable from ONE document.
- The document scorer sees ONLY your generated question and the document, not the original question, search query, or search reason above. Make the generated question self-contained by stating the relevant topic, entity, or information need explicitly.
- Do not use unresolved references such as "my question", "the user's question", or "the search query" unless you explicitly state the referenced information in the generated question. A broad but self-contained relevance question is allowed.
- Do not mention document IDs or ranks.
- Do not encode answers for individual documents.

Filter question:`);
});

void test("Qorrect prompt defaults missing search reason to an empty string", () => {
  const context = { originalQuestion: "Who designed it?", currentQuery: "engine" };
  const prompt = buildQorrectPrompt(context);
  assert.equal(prompt, buildQorrectPrompt({ ...context, searchReason: "" }));
  assert.ok(prompt.includes("Reason for current search:\n\nCurrent search query:\nengine"));
});

void test("Qorrect prompt preserves multiline and non-English context verbatim", () => {
  const prompt = buildQorrectPrompt({
    originalQuestion: '誰設計了 "analytical engine"?\n請提供來源。',
    currentQuery: "分析機\n設計者",
    searchReason: "確認人物\n與機器的關係",
  });
  assert.ok(prompt.includes('Original question:\n誰設計了 "analytical engine"?\n請提供來源。\n\n'));
  assert.ok(prompt.includes("Current search query:\n分析機\n設計者\n\n"));
  assert.ok(prompt.includes("Reason for current search:\n確認人物\n與機器的關係\n"));
});
