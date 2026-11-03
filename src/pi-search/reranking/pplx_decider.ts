import type { RerankCandidate, RerankScorer } from "./types";

const PPLX_AFFIRMATIVE_INSTRUCTION =
  "Given a yes/no filter question, judge whether the document provides sufficient evidence to answer Yes to that question. A document that supports No, merely mentions the topic, or provides insufficient evidence does not meet the requirement. Use only the document.";

export type PplxDeciderConfig = {
  /** API base including /v1, for example http://127.0.0.1:8002/v1. */
  baseUrl: string;
  model: string;
  /** Timeout per candidate request. */
  timeoutMs?: number;
  /** Maximum simultaneous requests per score call. Defaults to 4. */
  concurrency?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function createPplxDeciderScorer(
  config: PplxDeciderConfig,
  fetchImpl: typeof fetch = fetch,
): RerankScorer {
  const endpoint = new URL(`${config.baseUrl.replace(/\/+$/, "")}/decisions`);
  if (!["http:", "https:"].includes(endpoint.protocol)) {
    throw new Error("Decider baseUrl must use HTTP or HTTPS.");
  }
  if (!config.model.trim()) throw new Error("Decider model must be non-empty.");
  const timeoutMs = config.timeoutMs ?? 60_000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) {
    throw new Error("Decider timeoutMs must be a positive integer within the timer range.");
  }

  const concurrency = config.concurrency ?? 4;
  if (!Number.isSafeInteger(concurrency) || concurrency < 1) {
    throw new Error("Decider concurrency must be a positive integer.");
  }

  return {
    async score(criterion, candidates, signal) {
      signal?.throwIfAborted();
      if (!criterion.question.trim()) {
        throw new Error("Decider question must be non-empty.");
      }
      const scored: RerankCandidate[] = new Array(candidates.length);
      const controller = new AbortController();
      const workSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
      let nextIndex = 0;
      async function worker() {
        while (nextIndex < candidates.length) {
          workSignal.throwIfAborted();
          const index = nextIndex++;
          const candidate = candidates[index];
          const timeout = AbortSignal.timeout(timeoutMs);
          const response = await fetchImpl(endpoint, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: config.model,
              state: candidate.text,
              questions: {
                relevance: {
                  type: "noul",
                  instructions: `${PPLX_AFFIRMATIVE_INSTRUCTION}\n\nFilter question:\n${criterion.question}`,
                },
              },
            }),
            signal: AbortSignal.any([workSignal, timeout]),
          });
          if (!response.ok) {
            throw new Error(`Decider scoring failed: HTTP ${response.status}.`);
          }
          const payload: unknown = await response.json();
          if (!isRecord(payload) || payload.model !== config.model) {
            throw new Error("Invalid Decider response: model does not match the request.");
          }
          const answer = isRecord(payload.answers) ? payload.answers.relevance : undefined;
          if (
            !isRecord(answer) ||
            answer.type !== "noul" ||
            typeof answer.noul !== "number" ||
            !Number.isFinite(answer.noul) ||
            answer.noul < 0 ||
            answer.noul > 1
          ) {
            throw new Error(
              "Invalid Decider response: expected relevance.noul probability in [0, 1].",
            );
          }
          workSignal.throwIfAborted();
          scored[index] = { ...candidate, rerankScore: answer.noul };
        }
      }
      const workers = Array.from({ length: Math.min(concurrency, candidates.length) }, () =>
        worker(),
      );
      try {
        await Promise.all(workers);
      } catch (error) {
        controller.abort(error);
        await Promise.allSettled(workers);
        throw error;
      }
      return scored;
    },
  };
}
