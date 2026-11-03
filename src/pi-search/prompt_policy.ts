import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  formatPiSearchFinalResponseFormat,
  hasPiSearchOutputMode,
  type PiSearchOutputModes,
} from "./agent_prompt";

export const BENCHMARK_TIMEOUT_SECONDS = Number.parseFloat(process.env.TIMEOUT_SECONDS ?? "");
export const SUBMIT_NOW_TRIGGER_RATIO = 0.7;

/** Stop retrieval without changing the response contract requested by the benchmark. */
export function buildSubmitNowSteerMessage(
  outputModes: PiSearchOutputModes,
  rankedListCount?: number,
  rankedListDepth = 1000,
): string {
  const wantsRankedList = hasPiSearchOutputMode(outputModes, "ranked_list");
  return [
    "Time budget is nearly exhausted.",
    "Further retrieval is now blocked to preserve time for submission.",
    "Stop using tools immediately and submit your final response right now.",
    ...(wantsRankedList
      ? [rankedListCount
          ? `For the Ranked List section, list exactly ${rankedListCount} unique docids from documents already retrieved, in descending relevance. If fewer are available, submit all known valid docids; never invent IDs.`
          : `For the Ranked List section, list up to ${rankedListDepth} unique docids from documents already retrieved, in descending relevance; never invent IDs.`]
      : []),
    "Your final response must use exactly this format:",
    formatPiSearchFinalResponseFormat(outputModes),
    "Do not include other sections, commentary, citations or scores within the ranked list.",
  ].join("\n");
}

function stripSection(systemPrompt: string, header: string, endMarkers: string[]): string {
  const start = systemPrompt.indexOf(header);
  if (start === -1) {
    return systemPrompt;
  }

  let end = systemPrompt.length;
  for (const marker of endMarkers) {
    const markerIndex = systemPrompt.indexOf(marker, start + header.length);
    if (markerIndex !== -1 && markerIndex < end) {
      end = markerIndex;
    }
  }

  return `${systemPrompt.slice(0, start)}${systemPrompt.slice(end)}`;
}

export function stripBenchmarkIrrelevantSystemPromptSections(systemPrompt: string): string {
  const sectionEndMarkers = [
    "\n\n# Project Context\n\nProject-specific instructions and guidelines:\n\n",
    "\n\nThe following skills provide specialized instructions",
    "\nCurrent date:",
    "\nCurrent working directory:",
  ];

  let stripped = stripSection(
    systemPrompt,
    "\n\nPi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n",
    sectionEndMarkers,
  );
  stripped = stripSection(
    stripped,
    "\n\n# Project Context\n\nProject-specific instructions and guidelines:\n\n",
    [
      "\n\nThe following skills provide specialized instructions",
      "\nCurrent date:",
      "\nCurrent working directory:",
    ],
  );
  return stripped;
}

export function dumpPromptSnapshot(systemPrompt: string, prompt: string): void {
  if (process.env.PI_SEARCH_DUMP_PROMPTS !== "1") {
    return;
  }
  const outputDir = process.env.OUTPUT_DIR;
  if (!outputDir) {
    return;
  }

  const dumpDir = join(outputDir, "prompt-dumps");
  mkdirSync(dumpDir, { recursive: true });
  const dumpId = `${new Date().toISOString().replace(/[:.]/g, "-")}-pid${process.pid}`;
  writeFileSync(join(dumpDir, `${dumpId}.system.txt`), systemPrompt, "utf-8");
  writeFileSync(join(dumpDir, `${dumpId}.user.txt`), prompt, "utf-8");
}

export function getSubmitNowDelayMs(): number | null {
  if (!Number.isFinite(BENCHMARK_TIMEOUT_SECONDS) || BENCHMARK_TIMEOUT_SECONDS <= 0) {
    return null;
  }
  return Math.max(1, Math.floor(BENCHMARK_TIMEOUT_SECONDS * SUBMIT_NOW_TRIGGER_RATIO * 1000));
}
