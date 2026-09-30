/** Pure filter configuration. Call only when filtering is enabled. */
export type FilterQuestionSource = "agent" | "qorrect" | "general";
export type FilterPredictor = "qorrect" | "qwen";

export function filterPredictor(env = process.env): FilterPredictor {
  const explicit = env.PI_SEARCH_FILTER_PREDICTOR?.trim();
  if (explicit) {
    if (explicit === "qorrect" || explicit === "qwen") return explicit;
    throw new Error("PI_SEARCH_FILTER_PREDICTOR must be qorrect or qwen.");
  }
  const legacy = env.PI_SEARCH_FILTER_BACKEND?.trim() || "reranq";
  if (legacy === "reranq") return "qorrect";
  if (legacy === "qwen3-reranker") return "qwen";
  throw new Error("Unknown PI_SEARCH_FILTER_BACKEND; use PI_SEARCH_FILTER_PREDICTOR=qorrect|qwen.");
}

export function filterQuestionSource(env = process.env): FilterQuestionSource {
  // Preserve old Qwen relevance runs only when using the legacy backend switch.
  const fallback = !env.PI_SEARCH_FILTER_PREDICTOR?.trim() && env.PI_SEARCH_FILTER_BACKEND?.trim() === "qwen3-reranker" ? "general" : "agent";
  const source = env.PI_SEARCH_FILTER_QUESTION_SOURCE?.trim() || fallback;
  if (source === "agent" || source === "qorrect" || source === "general") return source;
  throw new Error("PI_SEARCH_FILTER_QUESTION_SOURCE must be agent, qorrect or general.");
}

export function filterThreshold(env = process.env): number {
  const raw = env.PI_SEARCH_FILTER_THRESHOLD?.trim();
  const value = raw ? Number(raw) : 0.5;
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("PI_SEARCH_FILTER_THRESHOLD must be a number in [0, 1].");
  return value;
}

export function generalFilterQuestion(query: string): string {
  return `Does this document contain information relevant to answering the search query "${query}"?`;
}
