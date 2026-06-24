import type { PiSearchBackendRuntime } from "./searcher/runtime";
import { PiSearchInvalidToolArgumentsError, PiSearchToolExecutionError } from "./protocol/errors";
import type {
  DirectReadDocumentParams,
  DirectSearchParams,
  PlainSearchParams,
  ReadDocumentParams,
  ReadSearchResultsParams,
  SelfBuiltSearchParams,
} from "./protocol/schemas";
import {
  buildReadSpillFileName,
  buildSearchSpillFileName,
  type ManagedTempSpillDir,
  truncateReadDocumentOutput,
  truncateSearchOutput,
} from "./spill";
import {
  buildSearchPage,
  formatSearchPageText,
  normalizePositiveInteger,
  SearchSessionStore,
} from "./search_cache";
import type {
  ReadDocumentDetails,
  ReadSearchResultsDetails,
  SearchDetails,
  ToolTimingBreakdown,
  SelfBuiltSearchDetails
} from "./tool_types";

import { spawn } from "node:child_process"; 

const SEARCH_QUERY_MODE = "plain";
const SEARCH_CACHE_K = 1000;
const SEARCH_FIRST_PAGE_LIMIT = 5;
const SEARCH_RESULTS_DEFAULT_LIMIT = 10;
const DIRECT_SEARCH_DEFAULT_LIMIT = 5;
const DIRECT_SEARCH_MAX_LIMIT = 100;

type SpillSequence = () => number;
type ToolExecutionContext = { cwd: string };

type ToolHandlerDeps = {
  backendRuntime: PiSearchBackendRuntime;
  searchStore: SearchSessionStore;
  spillDir: ManagedTempSpillDir;
  nextSpillSequence: SpillSequence;
};

type SelfSearchDeps = {
  searchScript: string;
  workSpace: string;
}

type RunCommandOptions = {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
};

type RunCommandResult = {
  stdout: string;
  stderr: string;
  code: number | null;
  signal: NodeJS.Signals | null;
};

function parseCommandString(input: string): { command: string; args: string[] } {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | null = null;
  let escaping = false;

  for (const char of input.trim()) {
    if (escaping) {
      current += char;
      escaping = false;
      continue;
    }

    if (char === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }

    if ((char === "'" || char === '"') && quote === null) {
      quote = char;
      continue;
    }

    if (char === quote) {
      quote = null;
      continue;
    }

    if (/\s/.test(char) && quote === null) {
      if (current.length > 0) {
        tokens.push(current);
        current = "";
      }
      continue;
    }

    current += char;
  }

  if (escaping) current += "\\";
  if (quote !== null) throw new Error(`Unclosed quote in command string: ${input}`);
  if (current.length > 0) tokens.push(current);
  if (tokens.length === 0) throw new Error("Command string is empty.");

  const [command, ...args] = tokens;
  return { command, args };
}

function runCommand(
  command: string,
  args: string[] = [],
  options: RunCommandOptions = {},
): Promise<RunCommandResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ? { ...process.env, ...options.env } : process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let timeout: NodeJS.Timeout | undefined;

    if (options.timeoutMs && options.timeoutMs > 0) {
      timeout = setTimeout(() => {
        child.kill("SIGTERM");
        setTimeout(() => {
          if (child.exitCode === null) child.kill("SIGKILL");
        }, 5_000).unref();
      }, options.timeoutMs);
    }

    child.stdout?.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr?.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      if (timeout) clearTimeout(timeout);
      reject(error);
    });

    child.on("close", (code, signal) => {
      if (timeout) clearTimeout(timeout);
      if (code === 0) {
        resolve({ stdout, stderr, code, signal });
        return;
      }
      reject(
        new Error(
          `Command failed: ${command} ${args.join(" ")}\n` +
            `exit code: ${code}${signal ? ` signal: ${signal}` : ""}\n` +
            `stderr:\n${stderr}`,
        ),
      );
    });
  });
}

function formatReadDocumentText(parsed: {
  docid: string;
  totalUnits?: number;
  returnedOffsetStart?: number;
  returnedOffsetEnd?: number;
  text: string;
  truncated: boolean;
  nextOffset?: number;
  limit: number;
}): string {
  const totalLines = parsed.totalUnits ?? 0;
  const returnedLineStart = parsed.returnedOffsetStart ?? 0;
  const returnedLineEnd = parsed.returnedOffsetEnd ?? 0;
  const lines = [
    `[docid=${parsed.docid} lines ${returnedLineStart}-${returnedLineEnd} of ${totalLines}]`,
    "",
    parsed.text,
  ];

  if (parsed.truncated && parsed.nextOffset) {
    lines.push("");
    lines.push(
      `[Document truncated. Continue with read_document({"docid":"${parsed.docid}","offset":${parsed.nextOffset},"limit":${parsed.limit}}).]`,
    );
  }

  return lines.join("\n").trim();
}

function formatDirectSearchText(args: {
  rawQuery: string;
  limit: number;
  hits: Array<{ docid: string; score?: number; title?: string | null; snippet?: string }>;
}): string {
  if (args.hits.length === 0) {
    return [`No hits returned.`, `Plain query: ${JSON.stringify(args.rawQuery)}`].join("\n");
  }

  const lines = [
    `Returned ${args.hits.length} hits from the Pyserini REST ranking.`,
    `Plain query: ${JSON.stringify(args.rawQuery)}`,
    `Requested hits: ${args.limit}`,
    "",
  ];
  for (const [index, hit] of args.hits.entries()) {
    const scoreText = typeof hit.score === "number" ? ` score=${hit.score.toFixed(4)}` : "";
    lines.push(`${index + 1}. docid=${hit.docid}${scoreText}`);
    if (hit.title) {
      lines.push(`   Title: ${hit.title}`);
    }
    if (hit.snippet) {
      lines.push(`   Excerpt: ${hit.snippet}`);
    } else {
      lines.push("   Excerpt: (No snippet available. Use read_document(docid) to inspect.)");
    }
    lines.push("");
  }
  lines.push("Use read_document(docid) to inspect a specific document before answering.");
  return lines.join("\n").trim();
}

function formatDirectReadDocumentText(args: {
  docid: string;
  title?: string | null;
  text: string;
}) {
  const lines = [`[docid=${args.docid} full document]`];
  if (args.title) {
    lines.push(`[title=${args.title}]`);
  }
  lines.push("", args.text);
  return lines.join("\n").trim();
}

function normalizeDirectSearchLimit(value: number | undefined): number {
  const parsed = normalizePositiveInteger(value, DIRECT_SEARCH_DEFAULT_LIMIT);
  return Math.min(parsed, DIRECT_SEARCH_MAX_LIMIT);
}

export async function executeSearchTool(
  params: PlainSearchParams,
  signal: AbortSignal | undefined,
  ctx: ToolExecutionContext,
  deps: ToolHandlerDeps,
) {
  const backend = deps.backendRuntime.getBackend(ctx.cwd);
  const rawQuery = String(params.query ?? "").trim();
  if (!rawQuery) {
    throw new PiSearchInvalidToolArgumentsError(
      "search arguments",
      "query must be a non-empty string.",
    );
  }
  const queryMode = SEARCH_QUERY_MODE;
  const response = await backend.search(
    {
      query: rawQuery,
      limit: SEARCH_CACHE_K,
    },
    signal,
  );

  const searchTiming: ToolTimingBreakdown = {
    searchRpcMs: response.timingMs?.request,
    serverInitMs: response.timingMs?.backendInit,
    serverUptimeMs: response.timingMs?.backendUptime,
  };
  const cached = deps.searchStore.createSearch(rawQuery, queryMode, response.hits);
  const page = buildSearchPage(cached, 1, SEARCH_FIRST_PAGE_LIMIT, searchTiming);
  const fullPageJson = JSON.stringify(page, null, 2);
  const rendered = truncateSearchOutput(
    deps.spillDir,
    buildSearchSpillFileName(page, deps.nextSpillSequence()),
    formatSearchPageText(page),
    fullPageJson,
  );

  return {
    content: [{ type: "text" as const, text: rendered.text }],
    details: {
      searchId: cached.searchId,
      rawQuery,
      queryMode: cached.queryMode,
      k: SEARCH_CACHE_K,
      totalCached: cached.results.length,
      returnedRankStart: page.returnedRankStart,
      returnedRankEnd: page.returnedRankEnd,
      nextOffset: page.nextOffset,
      retrievedDocids: cached.results.map((item) => item.docid),
      previewedDocids: page.results.map((item) => item.docid),
      timingMs: page.timingMs,
      truncation: rendered.truncation,
      fullOutputPath: rendered.fullOutputPath,
    } satisfies SearchDetails,
  };
}

export async function executeDirectSearchTool(
  params: DirectSearchParams,
  signal: AbortSignal | undefined,
  ctx: ToolExecutionContext,
  deps: ToolHandlerDeps,
) {
  const backend = deps.backendRuntime.getBackend(ctx.cwd);
  const rawQuery = String(params.query ?? "").trim();
  if (!rawQuery) {
    throw new PiSearchInvalidToolArgumentsError(
      "search arguments",
      "query must be a non-empty string.",
    );
  }
  const limit = normalizeDirectSearchLimit(params.hits);
  const response = await backend.search(
    {
      query: rawQuery,
      limit,
    },
    signal,
  );

  const renderedText = formatDirectSearchText({
    rawQuery,
    limit,
    hits: response.hits,
  });
  const rendered = truncateSearchOutput(
    deps.spillDir,
    `direct_search_${deps.nextSpillSequence()}.txt`,
    renderedText,
    JSON.stringify({ rawQuery, limit, hits: response.hits }, null, 2),
  );
  const docids = response.hits.map((hit) => hit.docid);
  return {
    content: [{ type: "text" as const, text: rendered.text }],
    details: {
      toolInterface: "pyserini-rest-2tool",
      rawQuery,
      queryMode: SEARCH_QUERY_MODE,
      k: limit,
      returnedRankStart: response.hits.length > 0 ? 1 : 0,
      returnedRankEnd: response.hits.length,
      retrievedDocids: docids,
      previewedDocids: docids,
      timingMs: {
        searchRpcMs: response.timingMs?.request,
        serverInitMs: response.timingMs?.backendInit,
        serverUptimeMs: response.timingMs?.backendUptime,
      },
      truncation: rendered.truncation,
      fullOutputPath: rendered.fullOutputPath,
    },
  };
}

export async function executeReadSearchResultsTool(
  params: ReadSearchResultsParams,
  _signal: AbortSignal | undefined,
  _ctx: ToolExecutionContext,
  deps: ToolHandlerDeps,
) {
  const offset = normalizePositiveInteger(params.offset, SEARCH_FIRST_PAGE_LIMIT + 1);
  const limit = normalizePositiveInteger(params.limit, SEARCH_RESULTS_DEFAULT_LIMIT);
  const cached = deps.searchStore.getSearch(params.search_id);
  if (!cached) {
    throw new PiSearchInvalidToolArgumentsError(
      "read_search_results arguments",
      `search_id '${params.search_id}' is unknown. Call search(...) first to create a result set.`,
    );
  }

  const page = buildSearchPage(cached, offset, limit);
  const fullPageJson = JSON.stringify(page, null, 2);
  const rendered = truncateSearchOutput(
    deps.spillDir,
    buildSearchSpillFileName(page, deps.nextSpillSequence()),
    formatSearchPageText(page),
    fullPageJson,
  );

  return {
    content: [{ type: "text" as const, text: rendered.text }],
    details: {
      searchId: cached.searchId,
      rawQuery: cached.rawQuery,
      queryMode: cached.queryMode,
      totalCached: cached.results.length,
      offset,
      limit,
      returnedRankStart: page.returnedRankStart,
      returnedRankEnd: page.returnedRankEnd,
      nextOffset: page.nextOffset,
      retrievedDocids: page.results.map((item) => item.docid),
      previewedDocids: page.results.map((item) => item.docid),
      timingMs: page.timingMs,
      truncation: rendered.truncation,
      fullOutputPath: rendered.fullOutputPath,
    } satisfies ReadSearchResultsDetails,
  };
}

export async function executeReadDocumentTool(
  params: ReadDocumentParams,
  signal: AbortSignal | undefined,
  ctx: ToolExecutionContext,
  deps: ToolHandlerDeps,
) {
  const backend = deps.backendRuntime.getBackend(ctx.cwd);
  const offset = normalizePositiveInteger(params.offset, 1);
  const limit = normalizePositiveInteger(params.limit, 200);
  const response = await backend.readDocument(
    {
      docid: params.docid,
      offset,
      limit,
    },
    signal,
  );

  const readTiming: ToolTimingBreakdown = {
    readDocumentRpcMs: response.timingMs?.request,
    serverInitMs: response.timingMs?.backendInit,
    serverUptimeMs: response.timingMs?.backendUptime,
  };
  if (!response.found) {
    throw new PiSearchToolExecutionError(
      "read_document",
      `docid '${params.docid}' was not found. Choose a docid returned by search(...) or read_search_results(...).`,
    );
  }

  const formatted = formatReadDocumentText({
    docid: response.docid,
    totalUnits: response.totalUnits,
    returnedOffsetStart: response.returnedOffsetStart,
    returnedOffsetEnd: response.returnedOffsetEnd,
    text: response.text,
    truncated: response.truncated,
    nextOffset: response.nextOffset,
    limit: response.limit,
  });
  const spillPayload = {
    docid: response.docid,
    offset: response.offset,
    limit: response.limit,
    returned_line_start: response.returnedOffsetStart,
    returned_line_end: response.returnedOffsetEnd,
  };
  const rendered = truncateReadDocumentOutput(
    deps.spillDir,
    buildReadSpillFileName(spillPayload, deps.nextSpillSequence()),
    formatted,
    formatted,
    {
      ...spillPayload,
      truncated: response.truncated,
      next_offset: response.nextOffset,
    },
  );

  return {
    content: [{ type: "text" as const, text: rendered.text }],
    details: {
      docid: params.docid,
      offset,
      limit,
      totalLines: response.totalUnits ?? 0,
      returnedLineStart: response.returnedOffsetStart ?? 0,
      returnedLineEnd: response.returnedOffsetEnd ?? 0,
      truncated: response.truncated,
      nextOffset: response.nextOffset,
      timingMs: readTiming,
      outputTruncation: rendered.truncation,
      fullOutputPath: rendered.fullOutputPath,
    } satisfies ReadDocumentDetails,
  };
}

export async function executeDirectReadDocumentTool(
  params: DirectReadDocumentParams,
  signal: AbortSignal | undefined,
  ctx: ToolExecutionContext,
  deps: ToolHandlerDeps,
) {
  const backend = deps.backendRuntime.getBackend(ctx.cwd);
  const response = await backend.readDocument({ docid: params.docid }, signal);

  const readTiming: ToolTimingBreakdown = {
    readDocumentRpcMs: response.timingMs?.request,
    serverInitMs: response.timingMs?.backendInit,
    serverUptimeMs: response.timingMs?.backendUptime,
  };
  if (!response.found) {
    throw new PiSearchToolExecutionError(
      "read_document",
      `docid '${params.docid}' was not found. Choose a docid returned by search(...).`,
    );
  }

  const formatted = formatDirectReadDocumentText({
    docid: response.docid,
    title: response.title,
    text: response.text,
  });
  const lineCount =
    response.totalUnits ?? (response.text ? response.text.split(/\r?\n/).length : 0);
  const rendered = truncateReadDocumentOutput(
    deps.spillDir,
    buildReadSpillFileName(
      {
        docid: response.docid,
        offset: 1,
        returned_line_start: lineCount > 0 ? 1 : undefined,
        returned_line_end: lineCount > 0 ? lineCount : undefined,
      },
      deps.nextSpillSequence(),
    ),
    formatted,
    formatted,
    {
      docid: response.docid,
      offset: 1,
      returned_line_start: lineCount > 0 ? 1 : undefined,
      returned_line_end: lineCount > 0 ? lineCount : undefined,
      truncated: false,
    },
  );

  return {
    content: [{ type: "text" as const, text: rendered.text }],
    details: {
      toolInterface: "pyserini-rest-2tool",
      docid: params.docid,
      totalLines: lineCount,
      truncated: false,
      timingMs: readTiming,
      outputTruncation: rendered.truncation,
      fullOutputPath: rendered.fullOutputPath,
    },
  };
}

export async function executeSelfBuiltSearchTool(
  params: SelfBuiltSearchParams,
  _signal: AbortSignal | undefined,
  _ctx: ToolExecutionContext,
  deps: SelfSearchDeps,
) {
  const { command, args } = parseCommandString(
    params.cmd
  );

  if (args.indexOf(deps.searchScript) == -1){
    throw new PiSearchToolExecutionError(
      "self-built search",
      `You can only run the provided python script ${deps.searchScript}`,
    );
  }

  if (args.indexOf(deps.workSpace) == -1) {
    throw new PiSearchToolExecutionError(
      "self-built search",
      `You can only use the workspace: ${deps.workSpace}`,
    );
  }

  const result = await runCommand(command, args);

  if (args.indexOf("--help") != -1 || args.indexOf("-h") != -1){
    return {
      content: [{ type: "text" as const, text: result.stdout }],
      details: {
        query: "",
        searchType: "help",
        matchedRubrics: {},
      } satisfies SelfBuiltSearchDetails,
    };
  }
  const response = JSON.parse(result.stdout) as {
    searchResponse: {
      query: string;
      searchType: string;
      docs: Array<{ id: string; content: string }>;
    };
    matchedRubrics: Record<string, string[]>;
  };
  let retrieved_docs = "";
  response.searchResponse.docs.forEach((element: { id: string; content: string }, idx: number) => {
    retrieved_docs += [
      `rank: ${idx + 1}`,
      `id: ${element.id}`,
      `content: ${element.content}`,
      "",
    ].join("\n");
  });
  
  return {
    content: [{ type: "text" as const, text: retrieved_docs || "No documents returned." }],
    details: {
      query: response.searchResponse.query,
      searchType: response.searchResponse.searchType,
      matchedRubrics: response.matchedRubrics,
    } satisfies SelfBuiltSearchDetails,
  };
}