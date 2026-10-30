import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  ANSWER_RESPONSE_FORMAT,
  RANKED_LIST_RESPONSE_FORMAT,
  parsePiSearchOutputModes,
} from "./agent_prompt";

export const BENCHMARK_TIMEOUT_SECONDS = Number.parseFloat(process.env.TIMEOUT_SECONDS ?? "");
export const SUBMIT_NOW_TRIGGER_RATIO = 0.7;
export function buildSubmitNowSteerMessage(env: NodeJS.ProcessEnv = process.env): string {
  const modes = parsePiSearchOutputModes(env.OUTPUT_MODE);
  const lines = [
    "Time budget is nearly exhausted.",
    "Further retrieval is now blocked to preserve time for submission.",
    "Stop using tools immediately and submit your final response right now using only the evidence and candidates already available.",
  ];
  if (modes.includes("ranked_list")) {
    const depth = Number(env.RANKED_LIST_DEPTH?.trim() || "1000");
    const count = env.RANKED_LIST_COUNT?.trim() ? Number(env.RANKED_LIST_COUNT) : undefined;
    if (
      !Number.isInteger(depth) ||
      depth <= 0 ||
      (count !== undefined && (!Number.isInteger(count) || count <= 0 || count > depth))
    ) {
      throw new Error(
        "Invalid ranked-list submission limits: require positive integers and count <= depth.",
      );
    }
    lines.push(
      count === undefined
        ? `Return as many relevant candidates as you can, up to ${depth} unique document ids.`
        : `The requested ranked list contains exactly ${count} unique document ids (maximum depth ${depth}). If you have fewer available candidates, submit those now; do not invent docids or resume retrieval to meet the count.`,
      "Deduplicate document ids and rank them by estimated relevance, most relevant first.",
      "Do not include scores, titles, snippets, citations, or explanations within the ranked list.",
    );
  }
  lines.push(
    "Your final response must contain exactly these sections in this order:",
    ...[
      modes.includes("answer") ? ANSWER_RESPONSE_FORMAT : undefined,
      modes.includes("ranked_list") ? RANKED_LIST_RESPONSE_FORMAT : undefined,
    ].filter((section): section is string => section !== undefined),
    "Do not include any text beyond the requested fields and ranked document lines.",
  );
  return lines.join("\n\n");
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
