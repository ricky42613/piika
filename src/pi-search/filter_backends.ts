import { filterPredictor, filterQuestionSource } from "./filter_config";

/** Model-specific prompt/label adapters. Transport and filtering policy are shared. */
export type SearchFilterBackend = "qwen3-reranker" | "jev" | "pplx-decider";

export const PPLX_GENERAL_INSTRUCTION =
  "Does the candidate document contain evidence that directly helps answer the search query?";
// Match reranq/decider_scorer.py AFFIRMATIVE_INSTRUCTION for Qorrect-generated questions.
export const PPLX_AFFIRMATIVE_INSTRUCTION =
  "Given a yes/no filter question, judge whether the document provides sufficient evidence to answer Yes to that question. A document that supports No, merely mentions the topic, or provides insufficient evidence does not meet the requirement. Use only the document.";

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
  scoringApi: "score" | "generative_scoring" | "systemone" | "vllm_prompt_logprobs" | "decider_score";
  probabilitySemantics: "normalized_yes_no_tokens" | "calibrated_decision_noul";
  yesLabels: string[];
  noLabels: string[];
  prefix: (question: string) => string;
  suffix: string;
};

export function resolveFilterBackend(env: NodeJS.ProcessEnv): FilterBackendProfile {
  const predictor = filterPredictor(env);
  const backend = predictor === "jev" ? "jev" : predictor === "pplx_decider" ? "pplx-decider" : "qwen3-reranker";
  const source = predictor === "jev" ? "general" : filterQuestionSource(env);
  const instruction =
    source === "general" ? QWEN3_FILTER_INSTRUCTION : QWEN3_AFFIRMATIVE_INSTRUCTION;
  const scoringApi =
    env.PI_SEARCH_FILTER_PREDICTOR_API?.trim() ||
    env.PI_SEARCH_FILTER_API?.trim() ||
    (backend === "jev" ? "systemone" : backend === "pplx-decider" ? "decider_score" : "score");
  if (
    scoringApi !== "score" &&
    scoringApi !== "generative_scoring" &&
    scoringApi !== "systemone" &&
    scoringApi !== "vllm_prompt_logprobs" &&
    scoringApi !== "decider_score"
  )
    throw new Error(`Unknown PI_SEARCH_FILTER_API: ${scoringApi}.`);
  if (backend === "jev" && scoringApi !== "systemone")
    throw new Error("JEV requires PI_SEARCH_FILTER_API=systemone.");
  if (backend === "pplx-decider" && scoringApi !== "decider_score")
    throw new Error("pplx_decider requires PI_SEARCH_FILTER_PREDICTOR_API=decider_score.");
  if (backend === "qwen3-reranker" && (scoringApi === "systemone" || scoringApi === "decider_score"))
    throw new Error("Qwen requires PI_SEARCH_FILTER_API=score|generative_scoring|vllm_prompt_logprobs.");
  const endpoint =
    env.PI_SEARCH_FILTER_PREDICTOR_BASE_URL?.trim() || env.PI_SEARCH_FILTER_BASE_URL?.trim();
  if (!endpoint) {
    throw new Error(`${backend} requires PI_SEARCH_FILTER_PREDICTOR_BASE_URL or PI_SEARCH_FILTER_BASE_URL.`);
  }
  const shared: Pick<FilterBackendProfile, "scoringApi" | "baseUrl" | "apiKey"> = {
    scoringApi,
    baseUrl: endpoint.replace(/\/$/, "").replace(/\/v1$/, ""),
    apiKey: env.PI_SEARCH_FILTER_PREDICTOR_API_KEY || env.PI_SEARCH_FILTER_API_KEY,
  };
  if (backend === "pplx-decider") {
    return {
      ...shared,
      backend,
      model: env.PI_SEARCH_FILTER_PREDICTOR_MODEL?.trim() || env.PI_SEARCH_FILTER_MODEL?.trim() || "pplx-decider-flash-v1.2",
      instruction: source === "general" ? PPLX_GENERAL_INSTRUCTION : PPLX_AFFIRMATIVE_INSTRUCTION,
      promptVersion: source === "general" ? "pplx-decider-query-relevance-v1" : "pplx-decider-affirmative-evidence-v2",
      probabilitySemantics: "calibrated_decision_noul",
      yesLabels: [],
      noLabels: [],
      prefix: (question) => question,
      suffix: "",
    };
  }
  if (backend === "jev") {
    return {
      ...shared,
      backend,
      model:
        env.PI_SEARCH_FILTER_PREDICTOR_MODEL?.trim() ||
        env.PI_SEARCH_FILTER_MODEL?.trim() ||
        "jev-latest",
      promptVersion: "jev-pointwise-noul-v1",
      probabilitySemantics: "normalized_yes_no_tokens",
      yesLabels: [],
      noLabels: [],
      prefix: (question) => question,
      suffix: "",
    };
  }
  return {
    ...shared,
    backend,
    model:
      env.PI_SEARCH_FILTER_PREDICTOR_MODEL?.trim() ||
      env.PI_SEARCH_FILTER_MODEL?.trim() ||
      "Qwen/Qwen3-Reranker-4B",
    instruction,
    promptVersion:
      source === "general" ? "qwen3-query-relevance-v1" : "qwen3-affirmative-evidence-v1",
    probabilitySemantics: "normalized_yes_no_tokens",
    yesLabels: scoringApi === "vllm_prompt_logprobs" ? [" Yes"] : ["yes"],
    noLabels: scoringApi === "vllm_prompt_logprobs" ? [" No"] : ["no"],
    prefix: (question) =>
      '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n' +
      `<Instruct>: ${instruction}\n<Query>: ${question}\n<Document>: `,
    suffix: QWEN3_FILTER_SUFFIX,
  };
}
