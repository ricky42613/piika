import type { RerankCriterion } from "./types";

export type CriterionContext = {
  currentQuery: string;
  originalQuestion: string;
  searchReason?: string;
};

export interface CriterionGenerator {
  generateQuestion(context: CriterionContext, signal?: AbortSignal): Promise<string>;
}

export async function buildCustomizedCriterion(
  context: CriterionContext,
  generator: CriterionGenerator,
  signal?: AbortSignal,
): Promise<RerankCriterion> {
  const currentQuery = context.currentQuery.trim();
  if (!currentQuery) {
    throw new Error("Cannot build a rerank criterion from an empty query.");
  }

  const question = (
    await generator.generateQuestion({ ...context, currentQuery }, signal)
  ).trim();
  if (!question) {
    throw new Error("Criterion generator returned an empty question.");
  }
  return { question };
}

/** Build the shared relevance question; provider-specific formatting belongs to the scorer. */
export function buildFixedCriterion(query: string): RerankCriterion {
  const currentQuery = query.trim();
  if (!currentQuery) {
    throw new Error("Cannot build a rerank criterion from an empty query.");
  }

  return {
    question: `Does this document provide information relevant to the search query "${currentQuery}"?`,
  };
}
