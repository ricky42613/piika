import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCustomizedCriterion,
  buildFixedCriterion,
} from "../../src/pi-search/reranking/criterion";

void test("fixed criterion uses the current query directly", () => {
  assert.deepEqual(buildFixedCriterion("  Who designed the analytical engine?  "), {
    question: "Who designed the analytical engine?",
  });
});

void test("fixed criterion preserves query content and is deterministic", () => {
  const query = '誰設計了 analytical engine?\nInclude "Charles Babbage".';
  const criterion = buildFixedCriterion(query);
  assert.equal(criterion.question, query);
  assert.deepEqual(buildFixedCriterion(query), criterion);
});

void test("fixed criterion rejects empty queries", () => {
  for (const query of ["", " \n\t "]) {
    assert.throws(() => buildFixedCriterion(query), /empty query/);
  }
});

void test("customized criterion forwards context and signal in one generator call", async () => {
  const context = Object.freeze({
    currentQuery: "  analytical engine  ",
    originalQuestion: "Who designed the analytical engine?",
    searchReason: "Find evidence identifying the designer",
  });
  const signal = new AbortController().signal;
  let calls = 0;
  const criterion = await buildCustomizedCriterion(context, {
    generateQuestion: async (received, receivedSignal) => {
      calls += 1;
      assert.deepEqual(received, { ...context, currentQuery: "analytical engine" });
      assert.strictEqual(receivedSignal, signal);
      return "  Does this document identify the designer of the analytical engine?\n";
    },
  }, signal);

  assert.deepEqual(criterion, {
    question: "Does this document identify the designer of the analytical engine?",
  });
  assert.equal(calls, 1);
  assert.equal(context.currentQuery, "  analytical engine  ");
});

void test("customized criterion supports context without a search reason", async () => {
  assert.deepEqual(
    await buildCustomizedCriterion({ currentQuery: "engine", originalQuestion: "Who designed the engine?" }, {
      generateQuestion: async () => "Does this document describe the engine?",
    }),
    { question: "Does this document describe the engine?" },
  );
});

void test("customized criterion rejects empty query without calling generator", async () => {
  let calls = 0;
  await assert.rejects(
    buildCustomizedCriterion({ currentQuery: " \n ", originalQuestion: "Who designed the engine?" }, {
      generateQuestion: async () => {
        calls += 1;
        return "unused";
      },
    }),
    /empty query/,
  );
  assert.equal(calls, 0);
});

void test("customized criterion rejects empty generated questions", async () => {
  await assert.rejects(
    buildCustomizedCriterion({ currentQuery: "engine", originalQuestion: "Who designed the engine?" }, {
      generateQuestion: async () => " \n\t ",
    }),
    /generator returned an empty question/,
  );
});

void test("customized criterion propagates generator errors", async () => {
  const error = new Error("generation failed");
  await assert.rejects(
    buildCustomizedCriterion({ currentQuery: "engine", originalQuestion: "Who designed the engine?" }, {
      generateQuestion: async () => { throw error; },
    }),
    (actual) => actual === error,
  );
});
