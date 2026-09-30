import assert from "node:assert/strict";
import test from "node:test";
import { createYesNoFilter } from "../../src/pi-search/yes_no_filter";
import {
  QWEN3_FILTER_INSTRUCTION,
  QWEN3_FILTER_SUFFIX,
  resolveFilterBackend,
} from "../../src/pi-search/filter_backends";

const env = {
  PI_SEARCH_FILTER_BACKEND: "qwen3-reranker",
  PI_SEARCH_FILTER_BASE_URL: "http://127.0.0.1:8152/v1",
  PI_SEARCH_FILTER_MODEL: "qwen3-reranker-4b",
};
function mockScore(
  options: {
    payload?: unknown;
    documentLength?: number;
    prefixLength?: number;
    status?: number;
  } = {},
) {
  const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
  const request = (async (url, init) => {
    assert.ok(typeof url === "string");
    const body = JSON.parse(init!.body as string) as Record<string, unknown>;
    calls.push({ url, body });
    assert.equal(body.model, "qwen3-reranker-4b");
    if (url.endsWith("/tokenize")) {
      assert.equal(body.add_special_tokens, false);
      const text = body.prompt as string;
      assert.ok(!["yes", "no", "Yes", "No"].includes(text));
      return Response.json({
        tokens: Array(
          text === QWEN3_FILTER_SUFFIX
            ? 2
            : text.startsWith("<|im_start|>")
              ? (options.prefixLength ?? 4)
              : (options.documentLength ?? 600),
        ).fill(100),
      });
    }
    assert.equal(url, "http://127.0.0.1:8152/score");
    if (options.status) return new Response("failed", { status: options.status });
    return Response.json(
      options.payload ?? { data: [{ index: 0, score: 0.9 }], usage: { prompt_tokens: 518 } },
    );
  }) as typeof fetch;
  return { calls, request };
}

void test("Qwen defaults to score; Reranq stays generative; invalid protocol combinations fail early", () => {
  assert.equal(resolveFilterBackend(env).scoringApi, "score");
  assert.equal(resolveFilterBackend({}).scoringApi, "generative_scoring");
  assert.throws(
    () => createYesNoFilter({ ...env, PI_SEARCH_FILTER_API: "rerank" }),
    /Unknown PI_SEARCH_FILTER_API/,
  );
  assert.throws(() => createYesNoFilter({ PI_SEARCH_FILTER_API: "score" }), /Reranq requires/);
});

void test("score sends the search query, relevance instruction, document-only budget and explicit activation", async () => {
  const { calls, request } = mockScore();
  const filter = createYesNoFilter(env, request);
  const result = await filter("Does the museum open on Sundays?", "document");
  assert.deepEqual(calls.at(-1)!.body, {
    model: "qwen3-reranker-4b",
    queries: "Does the museum open on Sundays?",
    documents: ["document"],
    instruction: QWEN3_FILTER_INSTRUCTION,
    max_tokens_per_query: 0,
    max_tokens_per_doc: 512,
    truncate_prompt_tokens: null,
    truncation_side: "right",
    use_activation: true,
  });
  assert.equal(calls.length, 4);
  assert.equal(result.keep, true);
  assert.equal(result.yesProbability, 0.9);
  assert.equal(result.backend, "qwen3-reranker");
  assert.equal(result.scoringApi, "score");
  assert.equal(result.promptVersion, "qwen3-query-relevance-v1");
  assert.equal(result.probabilitySemantics, "normalized_yes_no_tokens");
  assert.equal(result.promptTokens, 518);
  assert.equal(result.promptTokensSource, "server_usage");
  assert.equal(result.documentTokens, 512);
  assert.equal(result.documentTruncated, true);
  assert.equal(result.scoringRequests, 1);
  await filter("next?", "document");
  assert.equal(calls.length, 7);
});

void test("score strict threshold, missing optional usage, and short document accounting", async () => {
  for (const score of [0, 0.3, 0.5, 0.5001, 1]) {
    const result = await createYesNoFilter(
      env,
      mockScore({ documentLength: 10, payload: { data: [{ index: 0, score }] } }).request,
    )("q", "doc");
    assert.equal(result.keep, score > 0.5);
    assert.equal(result.documentTokens, 10);
    assert.equal(result.documentTruncated, false);
    assert.equal(result.promptTokens, undefined);
    assert.equal(result.promptTokensSource, undefined);
    assert.equal(result.yesLogprob, score === 0 ? null : Math.log(score));
  }
});

void test("score context budget shrinks document only and rejects oversized questions before scoring", async () => {
  const { calls, request } = mockScore({ prefixLength: 8100 });
  const result = await createYesNoFilter(env, request)("q", "doc");
  assert.equal(calls.at(-1)!.body.max_tokens_per_doc, 89);
  assert.equal(result.documentTokens, 89);
  const bad = mockScore({ prefixLength: 8190 });
  await assert.rejects(createYesNoFilter(env, bad.request)("q", "doc"), /too long/);
  assert.ok(bad.calls.every((c) => c.url.endsWith("/tokenize")));
});

void test("score rejects raw logits, malformed rows and invalid token usage without fallback", async () => {
  for (const payload of [
    {},
    { data: [] },
    { data: [{ index: 1, score: 0.9 }] },
    { data: [{ index: 0, score: 6.4 }] },
    { data: [{ index: 0, score: -0.1 }] },
    { data: [{ index: 0, score: "0.9" }] },
    { data: [{ index: 0, score: null }] },
    {
      data: [
        { index: 0, score: 0.8 },
        { index: 1, score: 0.2 },
      ],
    },
  ]) {
    await assert.rejects(
      createYesNoFilter(env, mockScore({ payload }).request)("q", "doc"),
      /invalid or incomplete score/,
    );
  }
  await assert.rejects(
    createYesNoFilter(
      env,
      mockScore({ payload: { data: [{ index: 0, score: 0.9 }], usage: { prompt_tokens: -1 } } })
        .request,
    )("q", "doc"),
    /invalid score prompt token usage/,
  );
  await assert.rejects(
    createYesNoFilter(env, mockScore({ status: 503 }).request)("q", "doc"),
    /HTTP 503/,
  );
  const mock = mockScore();
  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(createYesNoFilter(env, mock.request)("q", "doc", aborted.signal));
  await assert.rejects(createYesNoFilter(env, mock.request)(" ", "doc"), /non-empty/);
  assert.equal(mock.calls.length, 0);
});
