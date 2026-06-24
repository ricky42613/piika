import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";

import { formatPiSearchPromptWithSelfBuiltSearch } from "../pi-search/agent_prompt";
import { attachJsonlLineReader } from "../runtime/jsonl";
import { prepareIsolatedAgentDir } from "../runtime/pi_agent_dir";
import { parsePiEventJsonLine, type PiEvent } from "../runtime/pi_json_protocol";
import { startPiJsonProcess, startPiProcessTimeout } from "../runtime/pi_process";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_TIMEOUT_SECONDS = 900;
const DEFAULT_EXTENSION_PATH = "src/extensions/pi_search.ts";
const DEFAULT_MODEL = "openai-codex/gpt-5.4-mini";
const DEFAULT_THINKING = "medium";

type Args = {
  query: string;
  searchScript: string;
  timeoutSeconds: number;
};

function readEnv(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value ? value : undefined;
}

function parsePositiveNumber(value: string, label: string): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number; received ${value}`);
  }
  return parsed;
}

function printHelp(): void {
  console.log(`Run a query through pi using a self-built Python search script.

Usage:
  npm run run:prompt -- --query "question" --search-script path/to/search.py [--timeout-seconds 900]

Options:
  --query <text>               Question to answer.
  --search-script <path>       Python search script path included in the prompt.
  --timeout-seconds <seconds>  Hard timeout. Default: $TIMEOUT_SECONDS or ${DEFAULT_TIMEOUT_SECONDS}.
  -h, --help

Environment defaults:
  SEARCH_SCRIPT, TIMEOUT_SECONDS
`);
}

function parseArgs(argv: string[]): Args {
  let query: string | undefined;
  let searchScript = readEnv("SEARCH_SCRIPT");
  let timeoutSeconds = readEnv("TIMEOUT_SECONDS")
    ? parsePositiveNumber(readEnv("TIMEOUT_SECONDS") as string, "TIMEOUT_SECONDS")
    : DEFAULT_TIMEOUT_SECONDS;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    switch (arg) {
      case "--query":
        if (!next) throw new Error(`${arg} requires a value`);
        query = next;
        index += 1;
        break;
      case "--search-script":
        if (!next) throw new Error(`${arg} requires a value`);
        searchScript = next;
        index += 1;
        break;
      case "--timeout-seconds":
        if (!next) throw new Error(`${arg} requires a value`);
        timeoutSeconds = parsePositiveNumber(next, "timeoutSeconds");
        index += 1;
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!query) throw new Error("--query is required.");
  if (!searchScript) throw new Error("--search-script or SEARCH_SCRIPT is required.");

  return { query, searchScript, timeoutSeconds };
}

function resolveSearchScriptPath(path: string): string {
  if (isAbsolute(path)) return path;
  const candidates = [
    resolve(process.cwd(), path),
    resolve(REPO_ROOT, path),
    resolve(REPO_ROOT, "..", path),
  ];
  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0];
}

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const item of content) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    if (typeof record.text === "string") parts.push(record.text);
    if (typeof record.content === "string") parts.push(record.content);
  }
  return parts.join("\n").trim();
}

function assistantTextFromEvent(event: PiEvent): string | undefined {
  if (event.type !== "message_end") return undefined;
  const message = event.message;
  if (!message || typeof message !== "object") return undefined;
  const record = message as Record<string, unknown>;
  if (record.role !== "assistant") return undefined;
  const text = extractText(record.content);
  return text || undefined;
}

async function runPrompt(args: Args): Promise<string> {
  const sessionId = randomUUID();
  const outputDir = resolve("/mnt/data/home/ricky42613/polar-piika/piika/runs/", sessionId);
  console.log(`[run_prompt] output_dir=${outputDir}`);
  mkdirSync(outputDir, { recursive: true });
  const isolatedAgentDir = prepareIsolatedAgentDir(outputDir);
  const searchScriptPath = resolveSearchScriptPath(args.searchScript);
  const prompt = formatPiSearchPromptWithSelfBuiltSearch(
    args.query,
    searchScriptPath,
    outputDir,
  );

  const child = startPiJsonProcess({
    piBinary: readEnv("PI_BIN") ?? "pi",
    model: readEnv("MODEL") ?? DEFAULT_MODEL,
    thinking: readEnv("THINKING") ?? DEFAULT_THINKING,
    prompt,
    isolatedAgentDir,
    extensionPath: readEnv("EXTENSION") ?? DEFAULT_EXTENSION_PATH,
    cwd: REPO_ROOT,
    extraEnv: {
      PI_SEARCH_TOOL_INTERFACE: "self-built",
      RUN_PROMPT_SESSION_ID: sessionId,
      TIMEOUT_SECONDS: String(args.timeoutSeconds),
      OUTPUT_DIR: outputDir,
      WORKSPACE: outputDir,
      SEARCH_SCRIPT: searchScriptPath,
    },
  });

  const stdout = child.stdout;
  const stderr = child.stderr;
  if (!stdout || !stderr) throw new Error("Failed to start pi with piped stdout/stderr.");

  let finalAssistantText = "";
  let stderrTail = "";
  const timeout = startPiProcessTimeout({
    child,
    timeoutSeconds: args.timeoutSeconds,
    onTimeout: () => {
      console.error(`\n[run_prompt] timeout after ${args.timeoutSeconds}s; terminating pi`);
    },
  });

  const stopReading = attachJsonlLineReader(stdout, (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const event = parsePiEventJsonLine(trimmed, "pi JSON line");
    const text = assistantTextFromEvent(event);
    if (text) finalAssistantText = text;
  });

  stderr.on("data", (chunk) => {
    const text = chunk.toString();
    process.stderr.write(text);
    stderrTail = (stderrTail + text).slice(-4000);
  });

  return await new Promise((resolvePromise, reject) => {
    child.on("error", reject);
    child.on("close", (code) => {
      timeout.clear();
      stopReading();
      if (code !== 0) {
        reject(
          new Error(
            `pi exited with code ${code}${stderrTail ? `\nStderr tail:\n${stderrTail}` : ""}`,
          ),
        );
        return;
      }
      resolvePromise(finalAssistantText || "[No assistant final text found in pi JSON events]");
    });
  });
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  console.log(await runPrompt(args));
}

main().catch((error) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : inspect(error));
  process.exit(1);
});
