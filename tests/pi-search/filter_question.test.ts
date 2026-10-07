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
void test("Qorrect rollout-4 uses reason, training prompt and sampling defaults; excludes seen docs", async () => {
  const generate = createFilterQuestionGenerator(env, (async (url, init) => {
    assert.equal(url, "http://model:18080/v1/completions");
    assert.ok(init);
    assert.equal((init.headers as Record<string, string>).Authorization, "Bearer test");
    const bodyRaw = init.body;
    assert.equal(typeof bodyRaw, "string");
    const body = JSON.parse(bodyRaw as string);
    assert.deepEqual(body, {
      model: "ricky42613/qorrect-rollout4",
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

void test("rollout-4 prompt and chat template match the published checkpoint", () => {
  const question = buildFilterQuestionPrompt("Original question?", "search terms", "Find a missing fact", ["Seen document text"]);
  const seenQuestion = buildFilterQuestionPrompt("Original question?", "search terms", "Find a missing fact", ["Seen document text"], true);
  const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");
  assert.equal(sha256(question), "6d83efe70ec76e18381a9a3edaf432856e28f3cf80989d582dda878233b7d511");
  assert.equal(sha256(seenQuestion), "d50ba56962fba83f6537168d14c7d7128ec2412b662075fc7a3d00cd85f5e881");
  assert.equal(sha256(qwenChatPrompt(question)), "d85babab525e4ae3fec4a9a168ed878e61c61c30a293971991378ea51b672618");
});

void test("rollout-4 seen-docs prompt is opt-in, and generation settings can be overridden", async () => {
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
