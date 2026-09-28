/** Model-specific prompt/label adapters. Transport and filtering policy are shared. */
export type SearchFilterBackend = "reranq" | "qwen3-reranker";

export const CLASSIFICATION_SUFFIX =
  "\n\nAnswer the filter question using only this document.\nAnswer exactly Yes or No.\n\nAnswer:";

// Qwen's official Transformers template, with an instruction for affirmative
// evidence rather than generic relevance (a relevant document can answer No).
export const QWEN3_FILTER_INSTRUCTION =
  "Given a yes/no filter question, judge whether the document provides sufficient evidence to answer Yes to that question. A document that supports No, merely mentions the topic, or provides insufficient evidence does not meet the requirement. Use only the document.";
export const QWEN3_FILTER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

export type FilterBackendProfile = {
  backend: SearchFilterBackend;
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
  const backend = env.PI_SEARCH_FILTER_BACKEND?.trim() || "reranq";
  if (backend !== "reranq" && backend !== "qwen3-reranker") {
    throw new Error(`Unknown PI_SEARCH_FILTER_BACKEND: ${backend}; use reranq or qwen3-reranker.`);
  }
  const scoringApi =
    env.PI_SEARCH_FILTER_API?.trim() ||
    (backend === "qwen3-reranker" ? "score" : "generative_scoring");
  if (scoringApi !== "score" && scoringApi !== "generative_scoring")
    throw new Error(`Unknown PI_SEARCH_FILTER_API: ${scoringApi}.`);
  if (backend === "reranq" && scoringApi !== "generative_scoring")
    throw new Error("Reranq requires PI_SEARCH_FILTER_API=generative_scoring.");
  const endpoint = env.PI_SEARCH_FILTER_BASE_URL?.trim();
  if (backend === "qwen3-reranker" && !endpoint) {
    throw new Error(
      "qwen3-reranker requires PI_SEARCH_FILTER_BASE_URL for its deployed scoring service.",
    );
  }
  const shared: Pick<FilterBackendProfile, "scoringApi" | "baseUrl" | "apiKey"> = {
    scoringApi,
    baseUrl: (endpoint || "http://127.0.0.1:8150/v1").replace(/\/$/, "").replace(/\/v1$/, ""),
    apiKey: env.PI_SEARCH_FILTER_API_KEY || (backend === "reranq" ? env.RERANQ_API_KEY : undefined),
  };
  if (backend === "qwen3-reranker") {
    return {
      ...shared,
      backend,
      model: env.PI_SEARCH_FILTER_MODEL?.trim() || "Qwen/Qwen3-Reranker-4B",
      promptVersion: "qwen3-affirmative-evidence-v1",
      probabilitySemantics: "normalized_yes_no_tokens",
      yesLabels: ["yes"],
      noLabels: ["no"],
      requireSingleToken: true,
      prefix: (question) =>
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n' +
        `<Instruct>: ${QWEN3_FILTER_INSTRUCTION}\n<Query>: ${question}\n<Document>: `,
      suffix: QWEN3_FILTER_SUFFIX,
    };
  }
  return {
    ...shared,
    backend,
    model: env.PI_SEARCH_FILTER_MODEL?.trim() || "reranq-filter",
    promptVersion: "reranq-training-v1",
    probabilitySemantics: "normalized_yes_no_variants",
    yesLabels: ["Yes", " Yes"],
    noLabels: ["No", " No"],
    requireSingleToken: false,
    prefix: (question) => `Filter question:\n${question}\n\nDocument:\n`,
    suffix: CLASSIFICATION_SUFFIX,
  };
}
