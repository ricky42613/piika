import type { PiSearchRerankingConfig } from "../config";
import type { SearchRerankingOptions } from "../search_pipeline";
import type { PiSearchToolInterface } from "../tool_interface";
import { createPplxDeciderScorer } from "./pplx_decider";
import { createQorrectGenerator } from "./qorrect";
import { createQwenRerankerScorer } from "./qwen_reranker";

export type SearchRerankingContext = {
  originalQuestion?: string;
  searchReason?: string;
};

export type SearchReranker = {
  buildSearchOptions(context?: SearchRerankingContext): SearchRerankingOptions;
};

/** Resolve clients once; per-search context is supplied separately, never cached. */
export function createSearchReranker(
  config: PiSearchRerankingConfig | undefined,
  toolInterface: PiSearchToolInterface,
  dependencies: { env?: NodeJS.ProcessEnv; fetchImpl?: typeof fetch } = {},
): SearchReranker | undefined {
  if (!config || config.mode === "off") return undefined;
  if (toolInterface !== "pyserini-rest-2tool") {
    throw new Error("Reranking is supported only by the pyserini-rest-2tool interface.");
  }
  const fetchImpl = dependencies.fetchImpl ?? fetch;
  const scorer =
    config.scorer.provider === "qwen-reranker"
      ? createQwenRerankerScorer(config.scorer, fetchImpl)
      : createPplxDeciderScorer(config.scorer, fetchImpl, config.mode);

  if (config.mode === "fixed-template") {
    return {
      buildSearchOptions: () => ({ mode: "fixed-template", candidateK: config.candidateK, scorer }),
    };
  }

  const env = dependencies.env ?? process.env;
  const tokenEnv = config.generator.tokenEnv;
  const apiKey = tokenEnv ? env[tokenEnv]?.trim() : undefined;
  if (tokenEnv && !apiKey) {
    throw new Error(`Missing Qorrect API key in environment variable ${tokenEnv}.`);
  }
  const generator = createQorrectGenerator({ ...config.generator, apiKey }, fetchImpl);
  return {
    buildSearchOptions(context = {}) {
      if (!context.originalQuestion?.trim()) {
        throw new Error("Customized reranking requires the original question for this search.");
      }
      return {
        mode: "customized",
        candidateK: config.candidateK,
        scorer,
        generator,
        originalQuestion: context.originalQuestion,
        searchReason: context.searchReason,
      };
    },
  };
}
