import assert from "node:assert/strict";
import test from "node:test";
import { buildFilterQuestionPrompt, createFilterQuestionGenerator } from "../../src/pi-search/filter_question";
import { filterQuestionSource } from "../../src/pi-search/filter_config";
import { registerPiSearchExtension } from "../../src/pi-search/extension";
import { buildMockExtensionConfig } from "../../src/pi-search/config";
import { formatPiSearchPrompt } from "../../src/pi-search/agent_prompt";

void test("question source defaults to agent and is independent of predictor", () => {
  assert.equal(filterQuestionSource({}), "agent");
  assert.equal(filterQuestionSource({ PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect" }), "qorrect");
  assert.throws(() => filterQuestionSource({ PI_SEARCH_FILTER_QUESTION_SOURCE: "bad" }));
  assert.equal(filterQuestionSource({ PI_SEARCH_FILTER_PREDICTOR: "qwen", PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect" }), "qorrect");
  assert.throws(() => filterQuestionSource({ PI_SEARCH_FILTER_PREDICTOR: "qwen", PI_SEARCH_FILTER_QUESTION_SOURCE: "bad" }));
});

const env = { PI_SEARCH_ORIGINAL_QUESTION: "Original question?", PI_SEARCH_FILTER_GENERATOR_BASE_URL: "http://model:8150/v1", PI_SEARCH_FILTER_GENERATOR_API_KEY: "test", PI_SEARCH_FILTER_PREDICTOR: "qwen", PI_SEARCH_FILTER_PREDICTOR_BASE_URL: "http://not-generator", PI_SEARCH_FILTER_PREDICTOR_MODEL: "not-generator" };
void test("Qorrect generation uses training prompt, search query, and deterministic chat options", async () => {
  const generate = createFilterQuestionGenerator(env, (async (url, init) => {
    assert.equal(url, "http://model:8150/v1/chat/completions");
    assert.equal((init?.headers as Record<string, string>).Authorization, "Bearer test");
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body, {
      model: "reranq-filter", messages: [{ role: "user", content: buildFilterQuestionPrompt("Original question?", "search terms") }],
      temperature: 0, max_tokens: 64, chat_template_kwargs: { enable_thinking: false },
    });
    return Response.json({ choices: [{ finish_reason: "stop", message: { content: " Does it contain evidence? " } }] });
  }) as typeof fetch);
  assert.equal(await generate("search terms"), "Does it contain evidence?");
});

void test("generation rejects missing original question, failures, incomplete answers and cancellation", async () => {
  assert.throws(() => createFilterQuestionGenerator({}), /ORIGINAL_QUESTION/);
  for (const response of [
    new Response("bad", { status: 503 }),
    Response.json({ choices: [{ finish_reason: "length", message: { content: "partial" } }] }),
    Response.json({ choices: [{ finish_reason: "stop", message: { content: " " } }] }),
    Response.json({}),
  ]) {
    await assert.rejects(createFilterQuestionGenerator(env, (async () => response) as typeof fetch)("terms"));
  }
  const generate = createFilterQuestionGenerator(env, (async () => { throw new Error("must not request"); }) as typeof fetch);
  await assert.rejects(generate("terms", AbortSignal.abort()), /abort/i);
});

void test("Qorrect-generated mode removes question field and agent-writing instructions from schema and prompts", () => {
  const values = { PI_SEARCH_FILTER_ENABLED: "1", PI_SEARCH_FILTER_BACKEND: "reranq", PI_SEARCH_FILTER_QUESTION_SOURCE: "qorrect", PI_SEARCH_ORIGINAL_QUESTION: "Original?" };
  const old = { ...process.env };
  try {
    Object.assign(process.env, values);
    const tools: Array<{ parameters: { properties: Record<string, unknown> }; promptGuidelines?: string[] }> = [];
    registerPiSearchExtension({ on: () => {}, registerTool: (t: typeof tools[number]) => tools.push(t) } as never, {
      toolInterface: "pyserini-rest-2tool", resolveConfig: () => buildMockExtensionConfig({ documents: [] }),
    });
    assert.equal("filter_question" in tools[0].parameters.properties, false);
    assert.doesNotMatch(tools[0].promptGuidelines!.join(" "), /filter_question|supply.*yes\/no question/);
    assert.doesNotMatch(formatPiSearchPrompt("Original?", "plain_minimal", { toolInterface: "pyserini-rest-2tool" }), /filter_question|supply.*yes\/no question/);
  } finally {
    for (const key of Object.keys(values)) {
      if (old[key] === undefined) delete process.env[key]; else process.env[key] = old[key];
    }
  }
});
