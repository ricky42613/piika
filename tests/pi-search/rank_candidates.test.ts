import assert from "node:assert/strict";
import test from "node:test";

import { rankCandidates } from "../../src/pi-search/reranking/candidates";
import type { RerankCandidate } from "../../src/pi-search/reranking/types";

const candidates = Object.freeze([
  Object.freeze({ docid: "a", title: "A", text: "First document", retrievalScore: 10 }),
  Object.freeze({ docid: "b", text: "Second document", retrievalScore: 8 }),
  Object.freeze({ docid: "c", text: "Third document", retrievalScore: 3 }),
]);

void test("ranks scored candidates, preserves retrieval scores and original tie order without mutation", () => {
  const scored = Object.freeze([
    Object.freeze({ ...candidates[2], rerankScore: 0 }),
    Object.freeze({ ...candidates[0], rerankScore: -1 }),
    Object.freeze({ ...candidates[1], rerankScore: 0 }),
  ]);

  const result = rankCandidates(candidates, scored);
  assert.deepEqual(result, [
    { ...candidates[1], rerankScore: 0 },
    { ...candidates[2], rerankScore: 0 },
    { ...candidates[0], rerankScore: -1 },
  ]);
  assert.strictEqual(result[0], scored[2]);
  assert.deepEqual(candidates.map((candidate) => candidate.docid), ["a", "b", "c"]);
  assert.deepEqual(scored.map((candidate) => candidate.docid), ["c", "a", "b"]);
  assert.ok(candidates.every((candidate) => !("rerankScore" in candidate)));
});

void test("empty candidates and scorer output produce an empty ranking", () => {
  assert.deepEqual(rankCandidates([], []), []);
});

const invalidOutputs: Array<{
  name: string;
  scored: RerankCandidate[];
  error: RegExp;
}> = [
  {
    name: "missing candidates",
    scored: [{ ...candidates[0], rerankScore: 1 }],
    error: /Missing scored candidate docid: b/,
  },
  {
    name: "missing scores",
    scored: [candidates[0]],
    error: /Missing rerank score for docid: a/,
  },
  {
    name: "unknown docids",
    scored: [{ docid: "unknown", text: "Unknown", rerankScore: 1 }],
    error: /Unknown scored candidate docid: unknown/,
  },
  ...[NaN, Infinity, -Infinity].map((rerankScore) => ({
    name: `non-finite score ${rerankScore}`,
    scored: [{ ...candidates[0], rerankScore }],
    error: /Non-finite rerank score for docid: a/,
  })),
];

for (const { name, scored, error } of invalidOutputs) {
  void test(`rejects ${name}`, () => {
    assert.throws(() => rankCandidates(candidates, scored), error);
  });
}
