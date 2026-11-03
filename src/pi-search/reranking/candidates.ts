import type { RerankCandidate } from "./types";

/** Format model input at the scorer boundary; keep candidate fields unchanged. */
export function formatScoringDocument(candidate: Pick<RerankCandidate, "title" | "text">): string {
  return candidate.title ? `${candidate.title}\n${candidate.text}` : candidate.text;
}

/** Run before scoring. Keep the first occurrence of each docid in retrieval order. */
export function deduplicateCandidates<T extends { docid: string }>(candidates: readonly T[]): T[] {
  const seen = new Set<string>();
  return candidates.filter(({ docid }) => {
    if (seen.has(docid)) return false;
    seen.add(docid);
    return true;
  });
}

export type RankedCandidate = RerankCandidate & {
  rerankScore: number;
};

/**
 * Validate scores and candidate coverage, then rank.
 * Candidates must be deduplicated before scoring; the scorer returns one per input.
 */
export function rankCandidates(
  candidates: readonly RerankCandidate[],
  scoredCandidates: readonly RerankCandidate[],
): RankedCandidate[] {
  const originalRank = new Map<string, number>();
  for (const [index, candidate] of candidates.entries()) {
    originalRank.set(candidate.docid, index);
  }

  const seen = new Set<string>();
  const ranked: RankedCandidate[] = [];
  for (const candidate of scoredCandidates) {
    const { docid, rerankScore } = candidate;
    if (!originalRank.has(docid)) {
      throw new Error(`Unknown scored candidate docid: ${docid}`);
    }
    if (rerankScore === undefined) {
      throw new Error(`Missing rerank score for docid: ${docid}`);
    }
    if (!Number.isFinite(rerankScore)) {
      throw new Error(`Non-finite rerank score for docid: ${docid}`);
    }
    seen.add(docid);
    ranked.push(candidate as RankedCandidate);
  }

  for (const { docid } of candidates) {
    if (!seen.has(docid)) {
      throw new Error(`Missing scored candidate docid: ${docid}`);
    }
  }

  // Scorer output may be reordered; resolve ties using original retrieval order.
  return ranked.sort(
    (a, b) =>
      b.rerankScore - a.rerankScore || originalRank.get(a.docid)! - originalRank.get(b.docid)!,
  );
}
