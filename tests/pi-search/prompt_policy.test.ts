import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { buildSubmitNowSteerMessage, dumpPromptSnapshot } from "../../src/pi-search/prompt_policy";

void test("answer submission warning preserves answer fields only", () => {
  const message = buildSubmitNowSteerMessage({});
  assert.match(message, /Answer:\nExplanation:/);
  assert.match(message, /Exact Answer:/);
  assert.match(message, /Confidence:/);
  assert.doesNotMatch(message, /Ranked List:/);
});

void test("ranking submission warning preserves depth without requesting answer fields", () => {
  const message = buildSubmitNowSteerMessage({
    OUTPUT_MODE: "ranked_list",
    RANKED_LIST_DEPTH: "17",
  });
  assert.match(message, /up to 17 unique document ids/);
  assert.match(message, /Ranked List:\n1\. \{docid\}/);
  assert.doesNotMatch(message, /Explanation:|Exact Answer:|Confidence:|Answer:/);
  assert.match(buildSubmitNowSteerMessage({ OUTPUT_MODE: "ranked_list" }), /up to 1000/);
});

void test("combined submission warning preserves section order and exact count without resuming retrieval", () => {
  const message = buildSubmitNowSteerMessage({
    OUTPUT_MODE: "ranked_list+answer",
    RANKED_LIST_DEPTH: "20",
    RANKED_LIST_COUNT: "7",
  });
  assert.ok(message.indexOf("Answer:") < message.indexOf("Ranked List:"));
  assert.match(message, /exactly 7 unique document ids \(maximum depth 20\)/);
  assert.match(message, /do not invent docids or resume retrieval/);
  assert.match(message, /Stop using tools immediately/);
  assert.doesNotMatch(message, /Continue searching/);
});

void test("submission warning rejects invalid modes and ranking limits", () => {
  assert.throws(
    () => buildSubmitNowSteerMessage({ OUTPUT_MODE: "invalid" }),
    /Invalid output mode/,
  );
  for (const limits of [
    { RANKED_LIST_DEPTH: "0" },
    { RANKED_LIST_DEPTH: "1.5" },
    { RANKED_LIST_COUNT: "-1" },
    { RANKED_LIST_COUNT: "oops" },
    { RANKED_LIST_DEPTH: "5", RANKED_LIST_COUNT: "6" },
  ]) {
    assert.throws(
      () => buildSubmitNowSteerMessage({ OUTPUT_MODE: "ranked_list", ...limits }),
      /Invalid ranked-list submission limits/,
    );
  }
});

function withEnv(overrides: Record<string, string | undefined>, fn: () => void): void {
  const previous = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(overrides)) {
    previous.set(key, process.env[key]);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  try {
    fn();
  } finally {
    for (const [key, value] of previous) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  }
}

void test("dumpPromptSnapshot writes prompt dumps when PI_SEARCH_DUMP_PROMPTS=1", () => {
  const outputDir = mkdtempSync(join(tmpdir(), "pi-search-prompt-policy-"));

  try {
    withEnv(
      {
        OUTPUT_DIR: outputDir,
        PI_SEARCH_DUMP_PROMPTS: "1",
        PI_BM25_DUMP_PROMPTS: undefined,
      },
      () => {
        dumpPromptSnapshot("system prompt", "user prompt");
      },
    );

    const dumpDir = join(outputDir, "prompt-dumps");
    assert.equal(existsSync(dumpDir), true);

    const dumpFiles = readdirSync(dumpDir).sort();
    assert.equal(dumpFiles.length, 2);
    assert.match(dumpFiles[0], /\.system\.txt$|\.user\.txt$/);
    assert.match(dumpFiles[1], /\.system\.txt$|\.user\.txt$/);

    const contents = dumpFiles
      .map((fileName) => readFileSync(join(dumpDir, fileName), "utf8"))
      .sort();
    assert.deepEqual(contents, ["system prompt", "user prompt"]);
  } finally {
    rmSync(outputDir, { recursive: true, force: true });
  }
});

void test("dumpPromptSnapshot ignores the stale BM25-branded env name", () => {
  const outputDir = mkdtempSync(join(tmpdir(), "pi-search-prompt-policy-"));

  try {
    withEnv(
      {
        OUTPUT_DIR: outputDir,
        PI_SEARCH_DUMP_PROMPTS: undefined,
        PI_BM25_DUMP_PROMPTS: "1",
      },
      () => {
        dumpPromptSnapshot("system prompt", "user prompt");
      },
    );

    const dumpDir = join(outputDir, "prompt-dumps");
    assert.equal(existsSync(dumpDir), false);
  } finally {
    rmSync(outputDir, { recursive: true, force: true });
  }
});
