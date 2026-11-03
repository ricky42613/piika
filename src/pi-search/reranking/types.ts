/** Scoring input: the raw search query in fixed mode, or a Qorrect question in customized mode. */
export type RerankCriterion = {
  /** Existing field name retained for trace compatibility; may contain search keywords. */
  question: string;
};

/** Prepared scoring text, independent of backend snippets and document loading. */
export type RerankCandidate = {
  docid: string;
  title?: string;
  text: string;
  /** Original backend score, for example BM25. Preserve during reranking. */
  retrievalScore?: number;
  /** Finite value; higher means more relevant. Set by the scorer. */
  rerankScore?: number;
};

export interface RerankScorer {
  /**
   * Score all candidates against the same criterion.
   * Input docids must be unique. Return a new candidate with rerankScore set
   * for each input, preserving its other fields without mutating the input.
   * Output order is unspecified. Sorting belongs to the pipeline.
   * Rerank scores are not comparable across providers.
   */
  score(
    criterion: RerankCriterion,
    candidates: readonly RerankCandidate[],
    signal?: AbortSignal,
  ): Promise<RerankCandidate[]>;
}
