import assert from "node:assert/strict";
import test from "node:test";

import { deduplicateCandidates } from "../../src/pi-search/reranking/candidates";

void test("deduplicates by docid, keeping the first candidate and retrieval order without mutation", () => {
  const first = Object.freeze({ docid: "a", text: "First", retrievalScore: 10 });
  const second = Object.freeze({ docid: "b", text: "Second", retrievalScore: 8 });
  const duplicate = Object.freeze({ docid: "a", text: "Later", retrievalScore: 20 });
  const candidates = Object.freeze([first, second, duplicate, second]);

  const result = deduplicateCandidates(candidates);

  assert.deepEqual(result, [first, second]);
  assert.strictEqual(result[0], first);
  assert.equal(candidates.length, 4);
});

void test("preserves unique candidates and handles empty input", () => {
  const candidates = [{ docid: "a", text: "A" }, { docid: "b", text: "B" }];
  assert.deepEqual(deduplicateCandidates(candidates), candidates);
  assert.deepEqual(deduplicateCandidates([]), []);
});
