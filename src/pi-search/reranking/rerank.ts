import { deduplicateCandidates, rankCandidates, type RankedCandidate } from "./candidates";
import {
  buildCustomizedCriterion,
  buildFixedCriterion,
  type CriterionContext,
  type CriterionGenerator,
} from "./criterion";
import type { RerankCandidate, RerankCriterion, RerankScorer } from "./types";

export type RerankOptions = { scorer: RerankScorer } & (
  | { mode: "fixed-template"; context: Pick<CriterionContext, "currentQuery"> }
  | { mode: "customized"; context: CriterionContext; generator: CriterionGenerator }
);

export type RerankResult = {
  /** Null when there are no candidates and no criterion was generated. */
  criterion: RerankCriterion | null;
  candidates: RankedCandidate[];
};

/** Model-independent reranking; retrieval and output depth belong to the search pipeline. */
export async function rerankCandidates(
  candidates: readonly RerankCandidate[],
  options: RerankOptions,
  signal?: AbortSignal,
): Promise<RerankResult> {
  signal?.throwIfAborted();
  const unique = deduplicateCandidates(candidates);
  if (unique.length === 0) return { criterion: null, candidates: [] };

  const criterion =
    options.mode === "fixed-template"
      ? buildFixedCriterion(options.context.currentQuery)
      : await buildCustomizedCriterion(options.context, options.generator, signal);
  signal?.throwIfAborted();
  const scored = await options.scorer.score(criterion, unique, signal);
  signal?.throwIfAborted();
  return { criterion, candidates: rankCandidates(unique, scored) };
}
