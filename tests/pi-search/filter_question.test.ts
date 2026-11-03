import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  buildFilterQuestionPrompt,
  createFilterQuestionGenerator,
  qorrectSeenDocsEnabled,
  qwenChatPrompt,
} from "../../src/pi-search/filter_question";
import { filterQuestionSource } from "../../src/pi-search/filter_config";
import { registerPiSearchExtension } from "../../src/pi-search/extension";
import { buildMockExtensionConfig } from "../../src/pi-search/config";
import { formatPiSearchPrompt } from "../../src/pi-search/agent_prompt";

void test("question source defaults to agent and is independent of predictor", () => {
  assert.equal(filterQuestionSource({}), "agent");
  assert.equal(filterQuestionSource({ PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect" }), "qorrect");
  assert.throws(() => filterQuestionSource({ PI_SEARCH_FILTER_QUESTION_SOURCE: "bad" }));
  assert.equal(
    filterQuestionSource({
      PI_SEARCH_FILTER_PREDICTOR: "qwen",
      PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect",
    }),
    "qorrect",
  );
  assert.throws(() =>
    filterQuestionSource({
      PI_SEARCH_FILTER_PREDICTOR: "qwen",
      PI_SEARCH_FILTER_QUESTION_SOURCE: "bad",
    }),
  );
});

const env = {
  PI_SEARCH_ORIGINAL_QUESTION: "Original question?",
  PI_SEARCH_FILTER_GENERATOR_BASE_URL: "http://model:18080/v1",
  PI_SEARCH_FILTER_GENERATOR_API_KEY: "test",
  PI_SEARCH_FILTER_PREDICTOR: "qwen",
  PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "http://not-generator",
  PI_SEARCH_FILTER_PREDICTOR_MODEL: "not-generator",
};
void test("Qorrect uses reason, full-output training prompt and sampling defaults; excludes seen docs", async () => {
  const generate = createFilterQuestionGenerator(env, (async (url, init) => {
    assert.equal(url, "http://model:18080/v1/completions");
    assert.ok(init);
    assert.equal((init.headers as Record<string, string>).Authorization, "Bearer test");
    const bodyRaw = init.body;
    assert.equal(typeof bodyRaw, "string");
    const body = JSON.parse(bodyRaw as string);
    assert.deepEqual(body, {
      model: "qorrect-decider-fulloutput-checkpoint717",
      prompt: [qwenChatPrompt(buildFilterQuestionPrompt(
        "Original question?", "search terms", "Find a missing fact", ["Seen document text"],
      ))],
      max_tokens: 64,
      temperature: 1,
      top_p: 1.0,
      top_k: 0,
    });
    return Response.json({
      choices: [{ index: 0, finish_reason: "stop", text: " Does it contain evidence? " }],
    });
  }) as typeof fetch);
  assert.equal(await generate("search terms", "Find a missing fact", undefined, ["Seen document text"]), "Does it contain evidence?");
  assert.equal(qorrectSeenDocsEnabled({}), false);
  assert.doesNotMatch(buildFilterQuestionPrompt("q", "query", "reason", ["seen"]), /seen|Previously seen/);
  assert.match(buildFilterQuestionPrompt("q", "query", "reason"), /Reason for current search:\nreason/);
});

void test("Qorrect defaults to the full-output checkpoint endpoint and model without overriding explicit settings", async () => {
  const generate = createFilterQuestionGenerator({ PI_SEARCH_ORIGINAL_QUESTION: "Original question?" }, (async (url, init) => {
    assert.equal(url, "http://127.0.0.1:18086/v1/completions");
    const body = JSON.parse(init!.body as string);
    assert.equal(body.model, "qorrect-decider-fulloutput-checkpoint717");
    assert.match(body.prompt[0], /Make the generated question self-contained/);
    return Response.json({ choices: [{ text: "Does it answer the question?" }] });
  }) as typeof fetch);
  assert.equal(await generate("search terms", "reason"), "Does it answer the question?");
});

void test("prompt and chat template match reranq's frozen full-output constrained checkpoint", () => {
  // SHA-256 of reranq/outputs/grpo-base-decider-qwenjudge-fulloutput-constrained-epoch1-20261027T204501Z/prompts.py
  // build_question_prompt(example, enable_seen_docs=False/True), and its Qwen3.5 non-thinking chat template.
  const question = buildFilterQuestionPrompt("Original question?", "search terms", "Find a missing fact", ["Seen document text"]);
  const seenQuestion = buildFilterQuestionPrompt("Original question?", "search terms", "Find a missing fact", ["Seen document text"], true);
  const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");
  assert.equal(sha256(question), "017e081bdeb73b0b2057ed07df70724170b2af9548fe5f046962cbb609b37b52");
  assert.equal(sha256(seenQuestion), "b100e023ccd97f63e28614d74063dff5f77fe7bf3edaa47eead1bc4dfcc1408c");
  assert.equal(sha256(qwenChatPrompt(question)), "b0c967ddcfa880d1818a7f74b6b32c8584a1485533340ee31f7e697677822953");
  assert.match(question, /document scorer sees ONLY your generated question and the document/);
  assert.match(question, /Do not use unresolved references such as "my question"/);
});

void test("Qorrect seen-docs prompt is opt-in, and generation settings can be overridden", async () => {
  const optIn = { ...env, PI_SEARCH_FILTER_GENERATOR_INCLUDE_SEEN_DOCS: "1", PI_SEARCH_FILTER_GENERATOR_MODEL: "served-alias", PI_SEARCH_FILTER_GENERATOR_MAX_TOKENS: "32", PI_SEARCH_FILTER_GENERATOR_TEMPERATURE: "0.8" };
  const generate = createFilterQuestionGenerator(optIn, (async (_url, init) => {
    const body = JSON.parse(init!.body as string);
    assert.equal(body.model, "served-alias");
    assert.equal(body.max_tokens, 32);
    assert.equal(body.temperature, 0.8);
    assert.match(body.prompt[0], /Previously seen documents:\nseen text/);
    return Response.json({ choices: [{ text: "Does it answer the question?" }] });
  }) as typeof fetch);
  assert.equal(await generate("query", "reason", undefined, ["seen text"]), "Does it answer the question?");
  assert.throws(() => qorrectSeenDocsEnabled({ PI_SEARCH_FILTER_GENERATOR_INCLUDE_SEEN_DOCS: "oops" }));
});

void test("Qorrect passes the full generated completion without first-line clipping", async () => {
  const response = (async () => Response.json({ choices: [{ text: "  Does it answer?\nMore explanation and another question?  " }] })) as typeof fetch;
  assert.equal(await createFilterQuestionGenerator(env, response)("query", "reason"), "Does it answer?\nMore explanation and another question?");
});

void test("opt-in Qorrect identity pin checks the actual served model root before each generation", async () => {
  const pinned = { ...env, PI_SEARCH_FILTER_GENERATOR_MODEL: "qorrect-decider-general-inject", PI_SEARCH_FILTER_GENERATOR_EXPECTED_MODEL_ROOT: "/checkpoints/new-717" };
  const urls: string[] = [];
  const request = (async (url) => {
    urls.push(String(url));
    if (String(url).endsWith("/v1/models"))
      return Response.json({ data: [{ id: pinned.PI_SEARCH_FILTER_GENERATOR_MODEL, root: pinned.PI_SEARCH_FILTER_GENERATOR_EXPECTED_MODEL_ROOT }] });
    return Response.json({ choices: [{ text: "Does it answer?\n</think>" }] });
  }) as typeof fetch;
  const generate = createFilterQuestionGenerator(pinned, request);
  assert.equal(await generate("q", "r"), "Does it answer?\n</think>");
  assert.equal(await generate("q", "r"), "Does it answer?\n</think>");
  assert.deepEqual(urls, ["http://model:18080/v1/models", "http://model:18080/v1/completions", "http://model:18080/v1/models", "http://model:18080/v1/completions"]);
  let postCalled = false;
  const swapped = (async (url) => {
    if (String(url).endsWith("/v1/models")) return Response.json({ data: [{ id: pinned.PI_SEARCH_FILTER_GENERATOR_MODEL, root: "/checkpoints/old-717" }] });
    postCalled = true;
    throw new Error("should not generate from wrong model");
  }) as typeof fetch;
  await assert.rejects(createFilterQuestionGenerator(pinned, swapped)("q", "r"), /model identity changed/);
  assert.equal(postCalled, false);
});

void test("generation rejects missing original question, failures, empty answers and cancellation", async () => {
  assert.throws(() => createFilterQuestionGenerator({}), /ORIGINAL_QUESTION/);
  for (const response of [
    new Response("bad", { status: 503 }),
    Response.json({ choices: [{ finish_reason: "stop", text: " " }] }),
    Response.json({}),
  ]) {
    await assert.rejects(
      createFilterQuestionGenerator(env, (async () => response) as typeof fetch)("terms", "reason"),
    );
  }
  const generate = createFilterQuestionGenerator(env, (async () => {
    throw new Error("must not request");
  }) as typeof fetch);
  await assert.rejects(generate("terms", "reason", AbortSignal.abort()), /abort/i);
});

void test("Qorrect-generated mode removes question field and agent-writing instructions from schema and prompts", () => {
  const values = {
    PI_SEARCH_FILTER_ENABLED: "1",
    PI_SEARCH_FILTER_PREDICTOR: "qwen",
    PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "http://predictor",
    PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect",
    PI_SEARCH_ORIGINAL_QUESTION: "Original?",
  };
  const old = { ...process.env };
  try {
    Object.assign(process.env, values);
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
    assert.equal("filter_question" in tools[0].parameters.properties, false);
    assert.doesNotMatch(
      tools[0].promptGuidelines!.join(" "),
      /filter_question|supply.*yes\/no question/,
    );
    assert.doesNotMatch(
      formatPiSearchPrompt("Original?", "plain_minimal", { toolInterface: "pyserini-rest-2tool" }),
      /filter_question|supply.*yes\/no question/,
    );
  } finally {
    for (const key of Object.keys(values)) {
      if (old[key] === undefined) delete process.env[key];
      else process.env[key] = old[key];
    }
  }
});
