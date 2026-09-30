import { filterPredictor, filterQuestionSource, generalFilterQuestion } from "./filter_config";

/** Model-specific prompt/label adapters. Transport and filtering policy are shared. */
export type SearchFilterBackend = "reranq" | "qwen3-reranker";

export const CLASSIFICATION_SUFFIX =
  "\n\nAnswer the filter question using only this document.\nAnswer exactly Yes or No.\n\nAnswer:";

// Qwen's official Transformers template and default relevance instruction.
export const QWEN3_FILTER_INSTRUCTION =
  "Given a web search query, retrieve relevant passages that answer the query";
export const QWEN3_AFFIRMATIVE_INSTRUCTION =
  "Given a yes/no filter question, judge whether the document provides sufficient evidence to answer Yes to that question. A document that supports No, merely mentions the topic, or provides insufficient evidence does not meet the requirement. Use only the document.";
export const QWEN3_FILTER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

export type FilterBackendProfile = {
  backend: SearchFilterBackend;
  instruction?: string;
  model: string;
  baseUrl: string;
  apiKey?: string;
  promptVersion: string;
  scoringApi: "score" | "generative_scoring";
  probabilitySemantics: "normalized_yes_no_variants" | "normalized_yes_no_tokens";
  yesLabels: string[];
  noLabels: string[];
  requireSingleToken: boolean;
  prefix: (question: string) => string;
  suffix: string;
};

export function resolveFilterBackend(env: NodeJS.ProcessEnv): FilterBackendProfile {
  const backend = filterPredictor(env) === "qwen" ? "qwen3-reranker" : "reranq";
  const source = filterQuestionSource(env);
  const instruction = source === "general" ? QWEN3_FILTER_INSTRUCTION : QWEN3_AFFIRMATIVE_INSTRUCTION;
  const scoringApi =
    env.PI_SEARCH_FILTER_PREDICTOR_API?.trim() || env.PI_SEARCH_FILTER_API?.trim() ||
    (backend === "qwen3-reranker" ? "score" : "generative_scoring");
  if (scoringApi !== "score" && scoringApi !== "generative_scoring")
    throw new Error(`Unknown PI_SEARCH_FILTER_API: ${scoringApi}.`);
  if (backend === "reranq" && scoringApi !== "generative_scoring")
    throw new Error("Reranq requires PI_SEARCH_FILTER_API=generative_scoring.");
  const endpoint = env.PI_SEARCH_FILTER_PREDICTOR_BASE_URL?.trim() || env.PI_SEARCH_FILTER_BASE_URL?.trim();
  if (backend === "qwen3-reranker" && !endpoint) {
    throw new Error(
      "qwen3-reranker requires PI_SEARCH_FILTER_BASE_URL for its deployed scoring service.",
    );
  }
  const shared: Pick<FilterBackendProfile, "scoringApi" | "baseUrl" | "apiKey"> = {
    scoringApi,
    baseUrl: (endpoint || "http://127.0.0.1:8150/v1").replace(/\/$/, "").replace(/\/v1$/, ""),
    apiKey: env.PI_SEARCH_FILTER_PREDICTOR_API_KEY || env.PI_SEARCH_FILTER_API_KEY || (backend === "reranq" ? env.RERANQ_API_KEY : undefined),
  };
  if (backend === "qwen3-reranker") {
    return {
      ...shared,
      backend,
      model: env.PI_SEARCH_FILTER_PREDICTOR_MODEL?.trim() || env.PI_SEARCH_FILTER_MODEL?.trim() || "Qwen/Qwen3-Reranker-4B",
      instruction,
      promptVersion: source === "general" ? "qwen3-query-relevance-v1" : "qwen3-affirmative-evidence-v1",
      probabilitySemantics: "normalized_yes_no_tokens",
      yesLabels: ["yes"],
      noLabels: ["no"],
      requireSingleToken: true,
      prefix: (question) =>
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n' +
        `<Instruct>: ${instruction}\n<Query>: ${question}\n<Document>: `,
      suffix: QWEN3_FILTER_SUFFIX,
    };
  }
  return {
    ...shared,
    backend,
    model: env.PI_SEARCH_FILTER_PREDICTOR_MODEL?.trim() || env.PI_SEARCH_FILTER_MODEL?.trim() || "reranq-filter",
    promptVersion: source === "general" ? "qorrect-general-relevance-v1" : "reranq-training-v1",
    probabilitySemantics: "normalized_yes_no_variants",
    yesLabels: ["Yes", " Yes"],
    noLabels: ["No", " No"],
    requireSingleToken: false,
    prefix: (question) => `Filter question:\n${source === "general" ? generalFilterQuestion(question) : question}\n\nDocument:\n`,
    suffix: CLASSIFICATION_SUFFIX,
  };
}
