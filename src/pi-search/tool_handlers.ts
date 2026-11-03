import type { PiSearchBackendRuntime } from "./searcher/runtime";
import { PiSearchInvalidToolArgumentsError, PiSearchToolExecutionError } from "./protocol/errors";
import type {
  DirectReadDocumentParams,
  DirectSearchParams,
  PlainSearchParams,
  ReadDocumentParams,
  ReadSearchResultsParams,
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
} from "./tool_types";

import type { FilterPredictor, FilterQuestionSource, SearchFilterAction } from "./filter_config";
import type { PiSearchToolInterface } from "./tool_interface";
import type { FilterQuestionGenerator } from "./filter_question";
import { JEV_GENERAL_PROMPT, type DocumentFilter, type FilterScore } from "./yes_no_filter";
import type { DocumentReranker } from "./reranker";

const SEARCH_QUERY_MODE = "plain";
const SEARCH_CACHE_K = 1000;
const SEARCH_FIRST_PAGE_LIMIT = 5;
const SEARCH_RESULTS_DEFAULT_LIMIT = 10;
const DIRECT_SEARCH_DEFAULT_LIMIT = 5;
const DIRECT_SEARCH_MAX_LIMIT = 100;
const DEFAULT_REREANK_DEPTH = 50;
const MAX_QORRECT_SEEN_DOCS = 5;
const MAX_QORRECT_SEEN_DOC_CHARS = 4000;

type SpillSequence = () => number;
type ToolExecutionContext = { cwd: string };

type ToolHandlerDeps = {
  backendRuntime: PiSearchBackendRuntime;
  searchStore: SearchSessionStore;
  spillDir: ManagedTempSpillDir;
  nextSpillSequence: SpillSequence;
  documentFilter?: DocumentFilter;
  filterAction?: SearchFilterAction;
  documentReranker?: DocumentReranker;
  questionSource?: FilterQuestionSource;
  predictor?: FilterPredictor;
  filterThreshold?: number;
  generateFilterQuestion?: FilterQuestionGenerator;
  rerankDepth?: number;
  toolInterface?: PiSearchToolInterface;
  qorrectSeenClueDocids?: Set<string>;
  includeQorrectSeenDocs?: boolean;
};

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
  reranked?: boolean;
  rerankLabel?: string;
}): string {
  if (args.hits.length === 0) {
    return [`No hits returned.`, `Plain query: ${JSON.stringify(args.rawQuery)}`].join("\n");
  }

  const lines = [
    `Returned ${args.hits.length} hits from the ${args.rerankLabel ?? (args.reranked ? "Qwen3 reranked" : "Pyserini REST")} ranking.`,
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

function normalizeRereankDepth(value: number | undefined): number {
  const parsed = normalizePositiveInteger(value, DEFAULT_REREANK_DEPTH);
  return Math.min(parsed, DIRECT_SEARCH_MAX_LIMIT);
}

function parseClueDocids(value: string | undefined): string[] {
  if (typeof value !== "string") return [];
  const seen = new Set<string>();
  const docids: string[] = [];
  for (const raw of value.split(",")) {
    const docid = raw.trim();
    if (!docid || seen.has(docid)) continue;
    seen.add(docid);
    docids.push(docid);
  }
  return docids;
}

function renderSeenDocForQorrect(doc: { docid: string; title?: string | null; text: string }): string {
  const parts = [] as string[];
  if (doc.title?.trim()) parts.push(doc.title.trim());
  parts.push(doc.text);
  return parts.join("\n").slice(0, MAX_QORRECT_SEEN_DOC_CHARS);
}

async function loadQorrectSeenDocs(
  docids: readonly string[],
  backend: ReturnType<PiSearchBackendRuntime["getBackend"]>,
  signal: AbortSignal | undefined,
): Promise<string[]> {
  const docs: string[] = [];
  for (const docid of docids.slice(0, MAX_QORRECT_SEEN_DOCS)) {
    const response = await backend.readDocument({ docid }, signal);
    if (!response.found || !response.text) continue;
    docs.push(renderSeenDocForQorrect({ docid: response.docid, title: response.title, text: response.text }));
  }
  return docs;
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
  params: DirectSearchParams & {
    filter_question?: string;
    clue_docs?: string;
  },
  signal: AbortSignal | undefined,
  ctx: ToolExecutionContext,
  deps: ToolHandlerDeps,
) {
  const backend = deps.backendRuntime.getBackend(ctx.cwd);
  const hasClueDocsArg = typeof params.clue_docs === "string";
  const clueDocs = hasClueDocsArg ? params.clue_docs!.trim() : undefined;
  const rawQuery = String(params.query ?? "").trim();
  if (!rawQuery) {
    throw new PiSearchInvalidToolArgumentsError(
      "search arguments",
      "query must be a non-empty string.",
    );
  }
  const source = deps.documentFilter && deps.predictor === "jev"
    ? "jev"
    : (deps.questionSource ?? (deps.generateFilterQuestion ? "qorrect" : "agent"));
  if (deps.documentFilter && source === "qorrect" && !deps.generateFilterQuestion) {
    throw new Error("Qorrect question generator is not configured.");
  }
  const clueDocids = parseClueDocids(params.clue_docs);
  if (source === "qorrect" && deps.includeQorrectSeenDocs && deps.qorrectSeenClueDocids) {
    for (const docid of clueDocids) deps.qorrectSeenClueDocids.add(docid);
  }
  const qorrectSeenDocs =
    deps.documentFilter && source === "qorrect" && deps.includeQorrectSeenDocs && deps.qorrectSeenClueDocids?.size
      ? await loadQorrectSeenDocs(Array.from(deps.qorrectSeenClueDocids), backend, signal)
      : [];
  const filterQuestion =
    source === "jev" || source === "general"
      ? rawQuery
      : deps.documentFilter && source === "qorrect" && deps.generateFilterQuestion
        ? await deps.generateFilterQuestion(rawQuery, params.reason, signal, qorrectSeenDocs)
        : params.filter_question?.trim();
  if (deps.documentFilter && !filterQuestion) {
    throw new PiSearchInvalidToolArgumentsError(
      "search arguments",
      "filter_question must be a non-empty yes/no question.",
    );
  }
  const limit = normalizeDirectSearchLimit(params.hits);
  if (deps.documentFilter && deps.documentReranker)
    throw new Error("Search filtering and reranking cannot both be enabled.");
  const candidateLimit = deps.documentFilter || deps.documentReranker ? normalizeRereankDepth(deps.rerankDepth) : limit;
  const response = await backend.search(
    {
      query: rawQuery,
      limit: candidateLimit,
    },
    signal,
  );

  const filterScores: Array<FilterScore & { docid: string }> = [];
  const rerankScores: Array<{ docid: string; bm25Score?: number; score: number }> = [];
  const hits = [] as typeof response.hits;
  for (const hit of response.hits) {
    let documentText: string | undefined;
    if (deps.documentFilter || deps.documentReranker) {
      let offset = 1;
      const parts: string[] = [];
      while (true) {
        signal?.throwIfAborted();
        const doc = await backend.readDocument(
          offset === 1 ? { docid: hit.docid } : { docid: hit.docid, offset, limit: 10000 },
          signal,
        );
        if (!doc.found)
          throw new PiSearchToolExecutionError(
            "search",
            `Search candidate document '${hit.docid}' not found.`,
          );
        if (offset === 1 && doc.title) parts.push(doc.title);
        parts.push(doc.text);
        if (!doc.truncated) break;
        if (!doc.nextOffset || doc.nextOffset <= offset)
          throw new PiSearchToolExecutionError(
            "search",
            "Search candidate document pagination did not advance.",
          );
        offset = doc.nextOffset;
      }
      documentText = parts.join("\n");
    }
    if (deps.documentFilter) {
      const score = await deps.documentFilter(filterQuestion!, documentText!, signal);
      filterScores.push({ docid: hit.docid, ...score });
      if (deps.filterAction === "rerank") {
        const rankingScore = score.rerankScore ?? score.yesProbability;
        rerankScores.push({ docid: hit.docid, bm25Score: hit.score, score: rankingScore });
        hits.push({ ...hit, score: rankingScore });
        continue;
      }
      if (!score.keep) continue;
    }
    if (deps.documentReranker) {
      const score = await deps.documentReranker(rawQuery, documentText!, signal);
      rerankScores.push({ docid: hit.docid, bm25Score: hit.score, score });
      hits.push({ ...hit, score });
    } else hits.push(hit);
  }
  const rerankFromQuestion = !!deps.documentFilter && deps.filterAction === "rerank";
  if (deps.documentReranker || rerankFromQuestion) hits.sort((a, b) => b.score! - a.score!);
  const returnedHits = deps.documentFilter || deps.documentReranker ? hits.slice(0, limit) : hits;
  const rerankDetails = deps.documentReranker || rerankFromQuestion
    ? { query: rawQuery, ...(rerankFromQuestion ? {
          question: filterQuestion, inputSource: source, predictor: deps.predictor,
          yesNoScores: filterScores,
        } : {}), rereankDepth: candidateLimit, candidateCount: response.hits.length,
        returnedCount: returnedHits.length, scores: rerankScores }
    : undefined;
  const filterDetails = deps.documentFilter && !rerankFromQuestion
    ? {
        question: source === "jev" ? JEV_GENERAL_PROMPT : filterQuestion,
        inputSource: source,
        threshold: deps.filterThreshold ?? 0.5,
        requestedHits: limit,
        rereankDepth: candidateLimit,
        candidateCount: response.hits.length,
        keptCount: hits.length,
        returnedCount: returnedHits.length,
        scores: filterScores,
      }
    : undefined;
  const renderedText =
    (filterDetails
      ? `Filter kept ${hits.length} of ${response.hits.length} candidates (score > ${deps.filterThreshold ?? 0.5}); returning ${returnedHits.length} of requested ${limit}.\n`
      : "") +
    formatDirectSearchText({
      rawQuery,
      limit,
      hits: returnedHits,
      reranked: !!deps.documentReranker || rerankFromQuestion,
      rerankLabel: rerankFromQuestion && deps.predictor === "pplx_decider" ? "pplx-decider reranked" : undefined,
    });
  const rendered = truncateSearchOutput(
    deps.spillDir,
    `direct_search_${deps.nextSpillSequence()}.txt`,
    renderedText,
    JSON.stringify(
      { rawQuery, limit, candidateLimit, clueDocs, hits: returnedHits, filter: filterDetails, rerank: rerankDetails },
      null,
      2,
    ),
  );
  const docids = returnedHits.map((hit) => hit.docid);
  return {
    content: [{ type: "text" as const, text: rendered.text }],
    details: {
      toolInterface: deps.toolInterface ?? "pyserini-rest-2tool",
      rawQuery,
      queryMode: SEARCH_QUERY_MODE,
      ...(hasClueDocsArg ? { clueDocs } : {}),
      k: limit,
      returnedRankStart: returnedHits.length > 0 ? 1 : 0,
      returnedRankEnd: returnedHits.length,
      ...(filterDetails ? { filter: filterDetails } : {}),
      ...(rerankDetails ? { rerank: rerankDetails } : {}),
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
