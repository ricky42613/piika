import assert from "node:assert/strict";
import test from "node:test";

import { rerankCandidates } from "../../src/pi-search/reranking/rerank";
import type { RerankScorer } from "../../src/pi-search/reranking/types";

const first = Object.freeze({ docid: "a", text: "First", retrievalScore: 10 });
const second = Object.freeze({ docid: "b", text: "Second", retrievalScore: 8 });
const duplicate = Object.freeze({ docid: "a", text: "Duplicate", retrievalScore: 12 });
const candidates = Object.freeze([first, second, duplicate]);
const context = {
  originalQuestion: "Who designed the analytical engine?",
  currentQuery: "analytical engine designer",
  searchReason: "Identify the designer",
};
const unexpectedScorer: RerankScorer = {
  score: async () => {
    throw new Error("scorer must not be called");
  },
};

void test("fixed reranking deduplicates before scoring, preserves retrieval scores and returns sorted results", async () => {
  const signal = new AbortController().signal;
  let calls = 0;
  const result = await rerankCandidates(
    candidates,
    {
      mode: "fixed-template",
      context: { currentQuery: context.currentQuery },
      scorer: {
        score: async (criterion, received, receivedSignal) => {
          calls += 1;
          assert.equal(criterion.question, "analytical engine designer");
          assert.deepEqual(received, [first, second]);
          assert.strictEqual(received[0], first);
          assert.strictEqual(receivedSignal, signal);
          return received.map((candidate, index) => ({ ...candidate, rerankScore: index }));
        },
      },
    },
    signal,
  );
  assert.equal(calls, 1);
  assert.deepEqual(result.candidates, [
    { ...second, rerankScore: 1 },
    { ...first, rerankScore: 0 },
  ]);
  assert.ok(result.criterion);
  assert.ok(candidates.every((candidate) => !("rerankScore" in candidate)));
});

void test("customized reranking generates one shared criterion before scoring", async () => {
  const signal = new AbortController().signal;
  const events: string[] = [];
  const question = "Does this document identify\nthe designer of the analytical engine?";
  const result = await rerankCandidates(
    candidates,
    {
      mode: "customized",
      context,
      generator: {
        generateQuestion: async (received, receivedSignal) => {
          events.push("generate");
          assert.deepEqual(received, context);
          assert.strictEqual(receivedSignal, signal);
          return question;
        },
      },
      scorer: {
        score: async (criterion, received, receivedSignal) => {
          events.push("score");
          assert.deepEqual(criterion, { question });
          assert.strictEqual(receivedSignal, signal);
          assert.deepEqual(received, [first, second]);
          // Scorer order is unspecified; tied scores retain original retrieval order.
          return [...received].reverse().map((candidate) => ({ ...candidate, rerankScore: 0.5 }));
        },
      },
    },
    signal,
  );
  assert.deepEqual(events, ["generate", "score"]);
  assert.deepEqual(result.criterion, { question });
  assert.deepEqual(
    result.candidates.map((candidate) => candidate.docid),
    ["a", "b"],
  );
});

void test("empty candidates skip both generation and scoring", async () => {
  const result = await rerankCandidates([], {
    mode: "customized",
    context,
    scorer: unexpectedScorer,
    generator: {
      generateQuestion: async () => {
        throw new Error("must not generate");
      },
    },
  });
  assert.deepEqual(result, { criterion: null, candidates: [] });
});

void test("generator errors propagate without calling scorer", async () => {
  const error = new Error("generation failed");
  await assert.rejects(
    rerankCandidates(candidates, {
      mode: "customized",
      context,
      scorer: unexpectedScorer,
      generator: {
        generateQuestion: async () => {
          throw error;
        },
      },
    }),
    (actual) => actual === error,
  );
});

void test("scorer errors propagate without falling back to retrieval order", async () => {
  const error = new Error("scoring failed");
  await assert.rejects(
    rerankCandidates(candidates, {
      mode: "fixed-template",
      context,
      scorer: {
        score: async () => {
          throw error;
        },
      },
    }),
    (actual) => actual === error,
  );
});

void test("incomplete scorer output is rejected by ranking validation", async () => {
  await assert.rejects(
    rerankCandidates(candidates, {
      mode: "fixed-template",
      context,
      scorer: { score: async () => [{ ...first, rerankScore: 0.5 }] },
    }),
    /Missing scored candidate docid: b/,
  );
});

void test("already canceled reranking does not call any model", async () => {
  await assert.rejects(
    rerankCandidates(
      candidates,
      {
        mode: "fixed-template",
        context,
        scorer: unexpectedScorer,
      },
      AbortSignal.abort(),
    ),
    { name: "AbortError" },
  );
});

void test("cancellation after generation prevents scoring", async () => {
  const controller = new AbortController();
  await assert.rejects(
    rerankCandidates(
      candidates,
      {
        mode: "customized",
        context,
        scorer: unexpectedScorer,
        generator: {
          generateQuestion: async () => {
            controller.abort();
            return "Relevant?";
          },
        },
      },
      controller.signal,
    ),
    { name: "AbortError" },
  );
});

void test("cancellation after scoring prevents returning results", async () => {
  const controller = new AbortController();
  await assert.rejects(
    rerankCandidates(
      candidates,
      {
        mode: "fixed-template",
        context,
        scorer: {
          score: async (_criterion, received) => {
            controller.abort();
            return received.map((candidate) => ({ ...candidate, rerankScore: 1 }));
          },
        },
      },
      controller.signal,
    ),
    { name: "AbortError" },
  );
});
