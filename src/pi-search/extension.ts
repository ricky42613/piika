import { createFilterQuestionGenerator, qorrectSeenDocsEnabled } from "./filter_question";
import { parsePiSearchOutputModes } from "./agent_prompt";
import {
  filterPredictor,
  filterQuestionSource,
  filterThreshold,
  rereankDepth,
  searchFilterAction,
} from "./filter_config";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { resolvePiSearchExtensionConfigFromEnv, type PiSearchExtensionConfig } from "./config";
import {
  BENCHMARK_TIMEOUT_SECONDS,
  dumpPromptSnapshot,
  getSubmitNowDelayMs,
  stripBenchmarkIrrelevantSystemPromptSections,
  buildSubmitNowSteerMessage,
  SUBMIT_NOW_TRIGGER_RATIO,
} from "./prompt_policy";
import {
  DirectReadDocumentParamsSchema,
  ClueDocsDirectSearchParamsSchema,
  ClueDocsFilteredDirectSearchParamsSchema,
  DirectSearchParamsSchema,
  FilteredDirectSearchParamsSchema,
  PlainSearchParamsSchema,
  ReadDocumentParamsSchema,
  ReadSearchResultsParamsSchema,
} from "./protocol/schemas";
import { SearchSessionStore } from "./search_cache";
import {
  PiSearchBackendRuntime,
  type PiSearchBackendFactory,
  type PiSearchBackendRuntimeOptions,
} from "./searcher/runtime";
import { ManagedTempSpillDir } from "./spill";
import {
  executeDirectReadDocumentTool,
  executeDirectSearchTool,
  executeReadDocumentTool,
  executeReadSearchResultsTool,
  executeSearchTool,
} from "./tool_handlers";
import {
  isDirectTwoToolInterface,
  parsePiSearchToolInterface,
  usesSearchClueDocs,
  type PiSearchToolInterface,
} from "./tool_interface";

import { createYesNoFilter, searchFilterGuidance, isSearchFilterEnabled } from "./yes_no_filter";
import { createDocumentReranker, isSearchRerankEnabled } from "./reranker";

export type { PiSearchToolInterface } from "./tool_interface";

export type PiSearchExtensionOptions = {
  resolveConfig?: (env: NodeJS.ProcessEnv) => PiSearchExtensionConfig;
  backendRuntime?: PiSearchBackendRuntime;
  createBackend?: PiSearchBackendFactory;
  buildCacheKey?: PiSearchBackendRuntimeOptions["buildCacheKey"];
  spillDirPrefix?: string;
  toolInterface?: PiSearchToolInterface;
};

function resolveToolInterface(options: PiSearchExtensionOptions): PiSearchToolInterface {
  return parsePiSearchToolInterface(options.toolInterface ?? process.env.PI_SEARCH_TOOL_INTERFACE);
}

export function registerPiSearchExtension(
  pi: ExtensionAPI,
  options: PiSearchExtensionOptions = {},
): void {
  const extensionConfig =
    options.resolveConfig?.(process.env) ?? resolvePiSearchExtensionConfigFromEnv(process.env);
  const searchStore = new SearchSessionStore();
  const backendRuntime =
    options.backendRuntime ??
    new PiSearchBackendRuntime(extensionConfig, {
      buildCacheKey: options.buildCacheKey,
      createBackend: options.createBackend,
    });
  const spillDir = new ManagedTempSpillDir(options.spillDirPrefix ?? "pi-search-extension-");
  const toolInterface = resolveToolInterface(options);
  const filterEnabled = isSearchFilterEnabled(toolInterface);
  const rerankEnabled = isSearchRerankEnabled(toolInterface);
  if (filterEnabled && rerankEnabled) throw new Error("Search filtering and reranking cannot both be enabled.");
  const filterAction = filterEnabled ? searchFilterAction() : "filter";
  const directTwoTool = isDirectTwoToolInterface(toolInterface);
  const clueDocsEnabled = usesSearchClueDocs(toolInterface);
  const predictor = filterEnabled ? filterPredictor() : "qwen";
  const questionSource = filterEnabled
    ? predictor === "jev" ? "general" : filterQuestionSource()
    : "agent";
  const threshold = filterEnabled ? filterThreshold() : 0.5;
  const depth = filterEnabled || rerankEnabled ? rereankDepth() : undefined;
  const generatesFilterQuestion = filterEnabled && predictor !== "jev" && questionSource === "qorrect";
  const submitNowDelayMs = getSubmitNowDelayMs();
  const outputModes = parsePiSearchOutputModes(process.env.PI_SEARCH_OUTPUT_MODE);
  const parseOptionalPositiveInteger = (raw: string | undefined, name: string): number | undefined => {
    if (!raw?.trim()) return undefined;
    const value = Number(raw.trim());
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} must be a positive integer.`);
    return value;
  };
  const submitNowMessage = buildSubmitNowSteerMessage(
    outputModes,
    parseOptionalPositiveInteger(process.env.PI_SEARCH_RANKED_LIST_COUNT, "PI_SEARCH_RANKED_LIST_COUNT"),
    parseOptionalPositiveInteger(process.env.PI_SEARCH_RANKED_LIST_DEPTH, "PI_SEARCH_RANKED_LIST_DEPTH"),
  );
  let spillSequence = 0;
  let submitNowTimer: ReturnType<typeof setTimeout> | null = null;
  let submitNowMode = false;
  let promptSnapshotWritten = false;
  let spillCleanupRegistered = false;

  const cleanupSpillDir = () => {
    backendRuntime.dispose();
    spillDir.cleanup();
  };

  const registerSpillCleanup = () => {
    if (spillCleanupRegistered) return;
    spillCleanupRegistered = true;
    process.once("exit", cleanupSpillDir);
    process.once("SIGINT", cleanupSpillDir);
    process.once("SIGTERM", cleanupSpillDir);
  };

  const nextSpillSequence = (): number => {
    spillSequence += 1;
    return spillSequence;
  };

  const toolDeps = {
    backendRuntime,
    searchStore,
    spillDir,
    nextSpillSequence,
    documentFilter: filterEnabled ? createYesNoFilter() : undefined,
    filterAction,
    documentReranker: rerankEnabled ? createDocumentReranker() : undefined,
    questionSource,
    filterThreshold: threshold,
    generateFilterQuestion: generatesFilterQuestion ? createFilterQuestionGenerator() : undefined,
    rerankDepth: depth,
    toolInterface,
    qorrectSeenClueDocids: new Set<string>(),
    includeQorrectSeenDocs: generatesFilterQuestion && qorrectSeenDocsEnabled(),
    predictor,
  };
  const pyseriniRestPaginatedRead =
    directTwoTool &&
    extensionConfig.backend.kind === "pyserini-rest" &&
    extensionConfig.backend.readMode === "paginated";

  registerSpillCleanup();

  function clearSubmitNowTimer() {
    if (submitNowTimer !== null) {
      clearTimeout(submitNowTimer);
      submitNowTimer = null;
    }
  }

  pi.on("before_agent_start", async (event) => {
    const strippedSystemPrompt = stripBenchmarkIrrelevantSystemPromptSections(event.systemPrompt);
    if (!promptSnapshotWritten) {
      dumpPromptSnapshot(strippedSystemPrompt, event.prompt);
      promptSnapshotWritten = true;
    }
    if (strippedSystemPrompt === event.systemPrompt) {
      return;
    }
    return { systemPrompt: strippedSystemPrompt };
  });

  pi.on("agent_start", async (_event, ctx) => {
    clearSubmitNowTimer();
    submitNowMode = false;
    if (submitNowDelayMs === null) {
      return;
    }
    submitNowTimer = setTimeout(() => {
      if (submitNowMode || ctx.isIdle()) {
        return;
      }
      submitNowMode = true;
      try {
        console.error(
          `[pi-search] Time budget threshold reached at ${(submitNowDelayMs / 1000).toFixed(1)}s (${Math.round(SUBMIT_NOW_TRIGGER_RATIO * 100)}% of TIMEOUT_SECONDS=${BENCHMARK_TIMEOUT_SECONDS}); queueing submit-now steer and blocking further retrieval tools.`,
        );
        pi.sendUserMessage(submitNowMessage, { deliverAs: "steer" });
      } catch (error) {
        console.error(
          `[pi-search] Failed to queue submit-now steer: ${error instanceof Error ? error.message : String(error)}`,
        );
        submitNowMode = false;
      }
    }, submitNowDelayMs);
  });

  pi.on("tool_call", async (event) => {
    if (!submitNowMode) {
      return;
    }
    if (
      event.toolName === "search" ||
      event.toolName === "read_search_results" ||
      event.toolName === "read_document"
    ) {
      console.error(
        `[pi-search] Blocking ${event.toolName} after timeout steer; model must submit final answer now.`,
      );
      return {
        block: true,
        reason:
          "Time budget is nearly exhausted. Do not use more retrieval tools; submit your final answer right now.",
      };
    }
  });

  pi.on("agent_end", async () => {
    clearSubmitNowTimer();
    submitNowMode = false;
  });

  pi.on("session_shutdown", async () => {
    clearSubmitNowTimer();
    submitNowMode = false;
    cleanupSpillDir();
  });

  pi.registerTool({
    name: "search",
    label: "Search",
    description:
      directTwoTool
        ? "Search the configured backend and return ranked hits directly. The first argument must be reason, a brief rationale of at most 100 words."
        : "Search the configured pi-search backend using a raw query string. The first argument must be reason, a brief rationale of at most 100 words.",
    promptSnippet:
      directTwoTool
        ? "Always supply reason first, under 100 words. Use query for a concise lexical query. The tool returns ranked hits directly; inspect promising docids with read_document."
        : "Always supply reason first, under 100 words. Use query for a concise raw search string based on the original wording or one grounded refinement. The tool returns a search_id plus the first page of results.",
    promptGuidelines:
      directTwoTool
        ? [
            ...(filterEnabled && filterAction === "filter" ? [searchFilterGuidance()] : []),
            ...(filterEnabled && filterAction === "rerank" ? ["For each search, a Yes/No question is generated automatically from the original question and your current search query. Candidates are scored on their evidence for Yes and then reordered by score, highest first. No candidate is removed by a score threshold. The scorer reads the first 512 document tokens; read promising documents to verify them."] : []),
            ...(rerankEnabled ? ["Search retrieves the top REREANK_DEPTH BM25 candidates and uses Qwen3-Reranker-4B to reorder them by relevance to your search query. hits controls how many top reranked documents are returned. There is no Yes/No filtering or generated filter question."] : []),
            ...(clueDocsEnabled
              ? [
                  "Every search call must include clue_docs as a comma-separated string. Use an empty string when this query is based only on the original question or general reasoning. If this query was motivated by facts learned from read_document, list those source docids separated by commas.",
                ]
              : []),
            "Always provide reason first. Keep it specific and under 100 words.",
            "Use query as a short raw lexical query string, not a structured object and not raw Lucene syntax.",
            "The search result itself is the visible ranked list; there is no search_id or result-page browsing tool in this condition.",
            "Keep hits small by default. Increase hits only when the visible ranking is too shallow for the question.",
            "Open promising docids with read_document before answering.",
          ]
        : [
            "Always provide reason first. Keep it specific and under 100 words.",
            "Use query as a short raw lexical query string, not a structured object and not raw Lucene syntax.",
            "Start close to the original wording, then make grounded refinements only after browsing or reading.",
            "If the current ranking looks partially relevant, browse it before rewriting.",
            "After browsing a ranking that surfaces plausible candidates, inspect one with read_document(docid).",
          ],
    parameters:
      directTwoTool
        ? clueDocsEnabled
          ? filterEnabled && predictor !== "jev" && questionSource === "agent"
            ? ClueDocsFilteredDirectSearchParamsSchema
            : ClueDocsDirectSearchParamsSchema
          : filterEnabled && predictor !== "jev" && questionSource === "agent"
            ? FilteredDirectSearchParamsSchema
            : DirectSearchParamsSchema
        : PlainSearchParamsSchema,
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      return directTwoTool
        ? executeDirectSearchTool(params, signal, ctx, toolDeps)
        : executeSearchTool(params, signal, ctx, toolDeps);
    },
  });

  if (toolInterface === "pi-serini-3tool") {
    pi.registerTool({
      name: "read_search_results",
      label: "Read Search Results",
      description:
        "Read a cached search result set by search_id. Supports offset and limit for paginated browsing of ranked hits, similar to the built-in read tool. The first argument must be reason, a brief rationale of at most 100 words.",
      promptSnippet:
        "Always supply reason first, with a brief rationale of at most 100 words. Then read a cached search result set by search_id in paginated ranked-hit chunks using offset and limit.",
      promptGuidelines: [
        "Always provide reason as the first argument. Keep it specific and under 100 words.",
        "Use read_search_results to browse deeper ranks from an existing search result set before rewriting the query.",
        "If the current ranking looks partly relevant, inspect more ranks here rather than issuing another search immediately.",
        "When browse surfaces plausible candidate biographies, open one with read_document(docid).",
      ],
      parameters: ReadSearchResultsParamsSchema,
      async execute(_toolCallId, params, signal, _onUpdate, ctx) {
        return executeReadSearchResultsTool(params, signal, ctx, toolDeps);
      },
    });
  }

  pi.registerTool({
    name: "read_document",
    label: "Read Document",
    description:
      directTwoTool && !pyseriniRestPaginatedRead
        ? "Fetch a full document by docid from the configured backend. The first argument must be reason, a brief rationale of at most 100 words."
        : "Read a retrieved document by docid. Supports offset and limit for paginated line-based reading, similar to the built-in read tool. The first argument must be reason, a brief rationale of at most 100 words.",
    promptSnippet:
      directTwoTool && !pyseriniRestPaginatedRead
        ? "Always supply reason first, with a brief rationale of at most 100 words. Then fetch the full document by docid; no offset or limit is needed."
        : "Always supply reason first, with a brief rationale of at most 100 words. Then read a retrieved document by docid in paginated line-based chunks using offset and limit.",
    promptGuidelines:
      directTwoTool && !pyseriniRestPaginatedRead
        ? [
            "Always provide reason as the first argument. Keep it specific and under 100 words.",
            "Use read_document to verify evidence from a specific docid before answering.",
            "Do not provide offset or limit; this tool fetches the full document returned by the backend.",
          ]
        : [
            "Always provide reason as the first argument. Keep it specific and under 100 words.",
            "Use read_document to verify evidence from a specific docid before answering.",
            "Start with offset=1 and a moderate limit when first reading a document.",
            "If a document is truncated and still looks relevant, continue reading the same document with the suggested next offset before launching many new searches.",
          ],
    parameters:
      directTwoTool && !pyseriniRestPaginatedRead
        ? DirectReadDocumentParamsSchema
        : ReadDocumentParamsSchema,
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      return directTwoTool && !pyseriniRestPaginatedRead
        ? executeDirectReadDocumentTool(params, signal, ctx, toolDeps)
        : executeReadDocumentTool(params, signal, ctx, toolDeps);
    },
  });
}

export default function (pi: ExtensionAPI) {
  registerPiSearchExtension(pi);
}
