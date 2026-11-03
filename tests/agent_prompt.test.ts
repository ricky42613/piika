import assert from "node:assert/strict";
import test from "node:test";

import { formatPiSearchPrompt, parsePiSearchOutputModes } from "../src/pi-search/agent_prompt";

void test("rerank prompt does not tell the agent that low-scoring candidates are filtered", () => {
  const keys = ["PI_SEARCH_FILTER_ENABLED", "PI_SEARCH_FILTER_ACTION", "PI_SEARCH_FILTER_QUESTION_SOURCE"] as const;
  const previous = keys.map((key) => process.env[key]);
  try {
    process.env.PI_SEARCH_FILTER_ENABLED = "1";
    process.env.PI_SEARCH_FILTER_ACTION = "rerank";
    process.env.PI_SEARCH_FILTER_QUESTION_SOURCE = "qorrect";
    const prompt = formatPiSearchPrompt("alpha", "plain_minimal", { outputMode: "ranked_list" });
    assert.match(prompt, /reranks all of them by Yes probability/);
    assert.doesNotMatch(prompt, /Only candidates with score >|in their original order/);
  } finally {
    keys.forEach((key, index) => {
      const old = previous[index];
      if (old === undefined) delete process.env[key];
      else process.env[key] = old;
    });
  }
});

void test("ranked-list prompt requests an exact count when configured", () => {
  const prompt = formatPiSearchPrompt("alpha", "plain_minimal", {
    outputMode: "ranked_list",
    rankedListDepth: 100,
    rankedListCount: 30,
  });

  assert.match(prompt, /Return exactly 30 unique document ids/);
  assert.match(prompt, /do not return fewer than 30/);
});

void test("ranked-list prompt preserves up-to-depth behavior without an exact count", () => {
  const prompt = formatPiSearchPrompt("alpha", "plain_minimal", {
    outputMode: "ranked_list",
    rankedListDepth: 50,
  });

  assert.match(prompt, /up to 50 document ids/);
  assert.doesNotMatch(prompt, /Return exactly/);
});

void test("answer and ranked-list outputs compose from the two atomic modes", () => {
  const prompt = formatPiSearchPrompt("alpha", "plain_minimal", {
    outputModes: parsePiSearchOutputModes("answer+ranked_list"),
    rankedListCount: 3,
  });

  assert.match(prompt, /Answer output:/);
  assert.match(prompt, /Ranked-list output:/);
  assert.match(prompt, /Answer:\nExplanation:/);
  assert.match(prompt, /Ranked List:\n1\. \{docid\}/);
  assert.match(prompt, /same research pass for all outputs/);
});

void test("output mode composition deduplicates atoms", () => {
  assert.deepEqual(parsePiSearchOutputModes("answer,ranked_list+answer"), [
    "answer",
    "ranked_list",
  ]);
});

void test("prompt defaults to the direct two-tool workflow", () => {
  const prompt = formatPiSearchPrompt("alpha");

  assert.match(prompt, /ranked hits returned directly by search/);
  assert.doesNotMatch(prompt, /read_search_results/);
  assert.doesNotMatch(prompt, /offset=1/);
});

void test("clue-docs two-tool workflow documents clue_docs", () => {
  const prompt = formatPiSearchPrompt("alpha", "plain_minimal", {
    toolInterface: "pyserini-rest-2tool-clue-docs",
  });

  assert.match(prompt, /ranked hits returned directly by search/);
  assert.match(prompt, /Every search call must include clue_docs/);
  assert.doesNotMatch(prompt, /read_search_results/);
});

void test("three-tool workflow remains an explicit opt-in", () => {
  const prompt = formatPiSearchPrompt("alpha", "plain_minimal", {
    toolInterface: "pi-serini-3tool",
  });

  assert.match(prompt, /Browse the current ranking with read_search_results/);
  assert.match(prompt, /offset=1/);
});
