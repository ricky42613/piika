import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { buildSubmitNowSteerMessage, dumpPromptSnapshot } from "../../src/pi-search/prompt_policy";
import { formatPiSearchPrompt, formatPiSearchPromptWithSuppliedContext, type PiSearchOutputModes } from "../../src/pi-search/agent_prompt";

for (const modes of [["answer"], ["ranked_list"], ["answer", "ranked_list"]] as PiSearchOutputModes[]) {
  void test(`submit-now steer preserves the original ${modes.join("+")} response sections`, () => {
    const initial = formatPiSearchPrompt("Question?", "plain_minimal", {
      outputModes: modes, rankedListDepth: 30, rankedListCount: 30,
    });
    const steer = buildSubmitNowSteerMessage(modes, 30, 30);
    const expected = initial.split("Your final response must contain exactly these sections in this order:\n")[1]
      ?.split("\n\nDo not include any text")[0];
    assert.ok(expected);
    assert.ok(steer.includes(`Your final response must use exactly this format:\n${expected}`));
    if (modes.includes("ranked_list")) {
      assert.match(steer, /exactly 30 unique docids/);
      assert.match(steer, /never invent IDs/);
    } else {
      assert.doesNotMatch(steer, /unique docids/);
    }
    if (modes.length === 1 && modes[0] === "ranked_list") {
      assert.doesNotMatch(steer, /Explanation:|Exact Answer:|Confidence:|Answer:/);
      assert.match(steer, /Ranked List:\n1\. \{docid\}/);
    }
  });
}

void test("submit-now steer uses depth when no exact count and matches supplied-context format", () => {
  const modes: PiSearchOutputModes = ["ranked_list"];
  const steer = buildSubmitNowSteerMessage(modes, undefined, 100);
  const initial = formatPiSearchPromptWithSuppliedContext("Question?", "Evidence", { outputModes: modes });
  assert.match(steer, /up to 100 unique docids/);
  assert.match(steer, /Ranked List:\n1\. \{docid\}/);
  assert.match(initial, /Ranked List:\n1\. \{docid\}/);
  assert.doesNotMatch(steer, /Explanation:|Exact Answer:|Confidence:/);
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
