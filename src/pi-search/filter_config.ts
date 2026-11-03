/** Pure filter configuration. Call only when filtering is enabled. */
export type FilterQuestionSource = "agent" | "qorrect" | "general";
export type FilterPredictor = "qwen" | "jev" | "pplx_decider";
export type SearchFilterAction = "filter" | "rerank";

export function searchFilterAction(env = process.env): SearchFilterAction {
  const action = env.PI_SEARCH_FILTER_ACTION?.trim() || "filter";
  if (action === "filter" || action === "rerank") return action;
  throw new Error("PI_SEARCH_FILTER_ACTION must be filter or rerank.");
}

export function filterPredictor(env = process.env): FilterPredictor {
  if (env.PI_SEARCH_FILTER_BACKEND?.trim()) {
    throw new Error("PI_SEARCH_FILTER_BACKEND is no longer supported; use PI_SEARCH_FILTER_PREDICTOR=qwen|jev|pplx_decider.");
  }
  const predictor = env.PI_SEARCH_FILTER_PREDICTOR?.trim() || "qwen";
  if (predictor === "qwen" || predictor === "jev" || predictor === "pplx_decider") return predictor;
  throw new Error("PI_SEARCH_FILTER_PREDICTOR must be qwen, jev or pplx_decider; qorrect is a question source only.");
}

export function filterQuestionSource(env = process.env): FilterQuestionSource {
  const source = env.PI_SEARCH_FILTER_QUESTION_SOURCE?.trim() || "agent";
  if (source === "agent" || source === "qorrect" || source === "general") return source;
  throw new Error("PI_SEARCH_FILTER_QUESTION_SOURCE must be agent, qorrect or general.");
}

export function filterThreshold(env = process.env): number {
  const raw = env.PI_SEARCH_FILTER_THRESHOLD?.trim();
  const value = raw ? Number(raw) : 0.5;
  if (!Number.isFinite(value) || value < 0 || value > 1)
    throw new Error("PI_SEARCH_FILTER_THRESHOLD must be a number in [0, 1].");
  return value;
}

export function rereankDepth(env = process.env): number {
  const raw = env.REREANK_DEPTH?.trim();
  if (!raw) return 50;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 1)
    throw new Error("REREANK_DEPTH must be a positive number.");
  return Math.floor(value);
}
