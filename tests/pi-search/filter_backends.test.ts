import assert from "node:assert/strict";
import test from "node:test";
import {
  CLASSIFICATION_SUFFIX,
  QWEN3_FILTER_INSTRUCTION,
  QWEN3_FILTER_SUFFIX,
  resolveFilterBackend,
} from "../../src/pi-search/filter_backends";
import { createYesNoFilter } from "../../src/pi-search/yes_no_filter";

const qwenEnv = {
  PI_SEARCH_FILTER_BACKEND: "qwen3-reranker",
  PI_SEARCH_FILTER_API: "generative_scoring",
  PI_SEARCH_FILTER_BASE_URL: "http://qwen:8151/v1/",
};

function mockQwen(
  options: {
    score?: number;
    prefixLength?: number;
    documentLength?: number;
    tokens?: Record<string, unknown>;
    payload?: unknown;
  } = {},
) {
  const calls: Array<{ url: string; body: Record<string, unknown>; auth: string | null }> = [];
  const request = (async (url, init) => {
    assert.ok(typeof url === "string");
    const body = JSON.parse(init!.body as string) as Record<string, unknown>;
    calls.push({ url, body, auth: new Headers(init?.headers).get("Authorization") });
    assert.equal(body.add_special_tokens, false);
    if (url.endsWith("/tokenize")) {
      const text = body.prompt as string;
      const tokens =
        options.tokens?.[text] ??
        (text === "yes"
          ? [11]
          : text === "no"
            ? [12]
            : text === QWEN3_FILTER_SUFFIX
              ? [91, 92]
              : text.startsWith("<|im_start|>system")
                ? Array(options.prefixLength ?? 4).fill(90)
                : Array.from({ length: options.documentLength ?? 600 }, (_, i) => 100 + i));
      return Response.json({ tokens });
    }
    assert.equal(url, "http://qwen:8151/generative_scoring");
    assert.deepEqual(body.label_token_ids, [11, 12]);
    assert.equal(body.apply_softmax, true);
    assert.deepEqual(body.query, []);
    return Response.json(options.payload ?? { data: [{ index: 0, score: options.score ?? 0.8 }] });
  }) as typeof fetch;
  return { calls, request };
}

void test("backend selection defaults to unchanged Reranq; unknown backend and missing Qwen endpoint fail early", () => {
  const defaultProfile = resolveFilterBackend({});
  assert.deepEqual(defaultProfile.yesLabels, ["Yes", " Yes"]);
  assert.equal(defaultProfile.baseUrl, "http://127.0.0.1:8150");
  assert.equal(defaultProfile.model, "reranq-filter");
  assert.equal(defaultProfile.suffix, CLASSIFICATION_SUFFIX);
  assert.equal(defaultProfile.prefix("q"), "Filter question:\nq\n\nDocument:\n");
  assert.equal(resolveFilterBackend({ PI_SEARCH_FILTER_BACKEND: "reranq" }).backend, "reranq");
  assert.throws(
    () => createYesNoFilter({ PI_SEARCH_FILTER_BACKEND: "typo" }),
    /Unknown PI_SEARCH_FILTER_BACKEND/,
  );
  assert.throws(
    () => createYesNoFilter({ PI_SEARCH_FILTER_BACKEND: "qwen3-reranker" }),
    /requires PI_SEARCH_FILTER_BASE_URL/,
  );
});

void test("Qwen uses official chat boundaries, affirmative-evidence instruction, and only lowercase yes/no", async () => {
  const { calls, request } = mockQwen();
  const filter = createYesNoFilter(qwenEnv, request);
  const question = "Does this document say the museum opened in 1990?";
  const result = await filter(question, "document");
  assert.equal(result.backend, "qwen3-reranker");
  assert.equal(result.model, "Qwen/Qwen3-Reranker-4B");
  assert.equal(result.promptVersion, "qwen3-affirmative-evidence-v1");
  assert.equal(result.probabilitySemantics, "normalized_yes_no_tokens");
  assert.equal(result.yesProbability, 0.8);
  assert.equal(result.keep, true);
  assert.equal(result.documentTokens, 512);
  assert.equal(result.documentTruncated, true);
  assert.equal(result.promptTokens, 518);
  assert.equal(result.scoringRequests, 1);
  assert.deepEqual(
    calls.slice(0, 3).map((c) => c.body.prompt),
    ["yes", "no", QWEN3_FILTER_SUFFIX],
  );
  assert.equal(
    calls[3].body.prompt,
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n' +
      `<Instruct>: ${QWEN3_FILTER_INSTRUCTION}\n<Query>: ${question}\n<Document>: `,
  );
  const scoring = calls.at(-1)!;
  assert.deepEqual(scoring.body.items, [
    [90, 90, 90, 90, ...Array.from({ length: 512 }, (_, i) => 100 + i), 91, 92],
  ]);
  assert.equal(scoring.body.model, "Qwen/Qwen3-Reranker-4B");
  await filter("next?", "doc");
  assert.equal(calls.length, 9); // cached labels and suffix
});

void test("Qwen strict threshold rejects ties and negatives; short documents are not truncated", async () => {
  for (const score of [0, 0.2, 0.5, 0.5001, 1]) {
    const result = await createYesNoFilter(
      qwenEnv,
      mockQwen({ score, documentLength: 20 }).request,
    )("q", "doc");
    assert.equal(result.keep, score > 0.5);
    assert.equal(result.documentTokens, 20);
    assert.equal(result.documentTruncated, false);
    assert.equal(result.yesLogprob, score === 0 ? null : Math.log(score));
    assert.equal(result.noLogprob, score === 1 ? null : Math.log(1 - score));
  }
});

void test("Qwen preserves question and suffix under context pressure, rejects oversized questions", async () => {
  const result = await createYesNoFilter(qwenEnv, mockQwen({ prefixLength: 8100 }).request)(
    "q",
    "doc",
  );
  assert.equal(result.documentTokens, 89);
  assert.equal(result.promptTokens, 8191);
  const { calls, request } = mockQwen({ prefixLength: 8190 });
  await assert.rejects(createYesNoFilter(qwenEnv, request)("q", "doc"), /too long/);
  assert.ok(calls.every((c) => c.url.endsWith("/tokenize")));
});

void test("Qwen rejects invalid label tokens, collisions, tokenization and score responses", async () => {
  for (const tokens of [
    { yes: [11, 13] },
    { no: [] },
    { no: [11] },
    { yes: [-1] },
    { yes: "11" },
  ]) {
    await assert.rejects(createYesNoFilter(qwenEnv, mockQwen({ tokens }).request)("q", "doc"));
  }
  for (const payload of [
    {},
    { data: [] },
    { data: [{ index: 2, score: 0.9 }] },
    { data: [{ index: 0, score: -0.1 }] },
    { data: [{ index: 0, score: 1.1 }] },
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
      createYesNoFilter(qwenEnv, mockQwen({ payload }).request)("q", "doc"),
      /invalid or incomplete/,
    );
  }
});

void test("Qwen honors model/auth overrides without leaking the Reranq-specific API key", async () => {
  for (const key of [undefined, "qwen-secret"]) {
    const { calls, request } = mockQwen();
    await createYesNoFilter(
      {
        ...qwenEnv,
        PI_SEARCH_FILTER_MODEL: "served-alias",
        PI_SEARCH_FILTER_API_KEY: key,
        RERANQ_API_KEY: "reranq-secret",
      },
      request,
    )("q", "doc");
    assert.ok(calls.every((c) => c.auth === (key ? `Bearer ${key}` : null)));
    assert.ok(calls.every((c) => c.body.model === "served-alias"));
  }
});

void test("Qwen cancellation, blank question and HTTP failures never bypass filtering", async () => {
  const { calls, request } = mockQwen();
  const filter = createYesNoFilter(qwenEnv, request);
  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(filter("q", "doc", aborted.signal));
  await assert.rejects(filter("  ", "doc"), /non-empty/);
  assert.equal(calls.length, 0);
  await assert.rejects(
    createYesNoFilter(
      qwenEnv,
      (async () => new Response("failed", { status: 503 })) as typeof fetch,
    )("q", "doc"),
    /HTTP 503/,
  );
});
