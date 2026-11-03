import type { RerankCandidate, RerankScorer } from "./types";
import { formatScoringDocument } from "./candidates";

export type QwenRerankerConfig = {
  /** Server root, without /v1; for example http://127.0.0.1:8003. */
  baseUrl: string;
  model: string;
  timeoutMs?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function createQwenRerankerScorer(
  config: QwenRerankerConfig,
  fetchImpl: typeof fetch = fetch,
): RerankScorer {
  const endpoint = new URL(`${config.baseUrl.replace(/\/+$/, "")}/score`);
  if (!["http:", "https:"].includes(endpoint.protocol)) {
    throw new Error("Qwen reranker baseUrl must use HTTP or HTTPS.");
  }
  if (!config.model.trim()) throw new Error("Qwen reranker model must be non-empty.");
  const timeoutMs = config.timeoutMs ?? 60_000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) {
    throw new Error("Qwen reranker timeoutMs must be a positive integer within the timer range.");
  }

  return {
    async score(criterion, candidates, signal) {
      signal?.throwIfAborted();
      if (!criterion.question.trim()) {
        throw new Error("Qwen reranker question must be non-empty.");
      }
      if (candidates.length === 0) return [];
      const timeout = AbortSignal.timeout(timeoutMs);
      const response = await fetchImpl(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: config.model,
          queries: criterion.question,
          documents: candidates.map(formatScoringDocument),
          instruction: "Determine whether the document satisfies the requirement in the query.",
          use_activation: true,
        }),
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
      if (!response.ok) {
        throw new Error(`Qwen reranker scoring failed: HTTP ${response.status}.`);
      }
      const payload: unknown = await response.json();
      if (!isRecord(payload) || payload.model !== config.model) {
        throw new Error("Invalid Qwen reranker response: model does not match the request.");
      }
      if (!Array.isArray(payload.data) || payload.data.length !== candidates.length) {
        throw new Error("Invalid Qwen reranker response: expected one score per candidate.");
      }
      const scored: RerankCandidate[] = new Array(candidates.length);
      const seen = new Set<number>();
      for (const entry of payload.data) {
        if (
          !isRecord(entry) ||
          typeof entry.index !== "number" ||
          !Number.isInteger(entry.index) ||
          entry.index < 0 ||
          entry.index >= candidates.length ||
          seen.has(entry.index) ||
          typeof entry.score !== "number" ||
          !Number.isFinite(entry.score)
        ) {
          throw new Error(
            "Invalid Qwen reranker response: expected unique valid indices and finite scores.",
          );
        }
        seen.add(entry.index);
        scored[entry.index] = { ...candidates[entry.index], rerankScore: entry.score };
      }
      signal?.throwIfAborted();
      return scored;
    },
  };
}
