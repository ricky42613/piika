import { deduplicateCandidates } from "./reranking/candidates";
import type { CriterionGenerator } from "./reranking/criterion";
import { prepareRerankCandidates } from "./reranking/document_loader";
import { rerankCandidates, type RerankOptions } from "./reranking/rerank";
import type { RerankCriterion, RerankScorer } from "./reranking/types";
import type { PiSearchBackend } from "./searcher/contract/interface";
import type {
  SearchBackendSearchHit,
  SearchBackendSearchRequest,
  SearchBackendSearchResponse,
} from "./searcher/contract/types";

export type SearchRerankingOptions = { candidateK: number; scorer: RerankScorer } & (
  | { mode: "fixed-template" }
  | {
      mode: "customized";
      generator: CriterionGenerator;
      originalQuestion: string;
      searchReason?: string;
    }
);

export type SearchPipelineResponse = Omit<SearchBackendSearchResponse, "hits"> & {
  hits: Array<SearchBackendSearchHit & { rerankScore?: number }>;
  reranking?: {
    criterion: RerankCriterion | null;
    /** Unique candidates in original retrieval order, including those not returned. */
    candidateDocids: string[];
    ranking: Array<{ docid: string; retrievalScore?: number; rerankScore: number }>;
  };
};

/** Direct search pipeline for the 2-tool interface; no result-page browsing. */
export async function executeSearchPipeline(
  backend: PiSearchBackend,
  request: Pick<SearchBackendSearchRequest, "query" | "limit">,
  signal?: AbortSignal,
  reranking?: SearchRerankingOptions,
): Promise<SearchPipelineResponse> {
  if (!reranking) return backend.search(request, signal);
  if (!Number.isSafeInteger(reranking.candidateK) || reranking.candidateK < 1) {
    throw new Error("Rerank candidateK must be a positive integer.");
  }
  signal?.throwIfAborted();
  const response = await backend.search(
    { ...request, limit: Math.max(reranking.candidateK, request.limit) },
    signal,
  );
  const hits = deduplicateCandidates(response.hits);
  const candidates = await prepareRerankCandidates(backend, hits, signal);
  const options: RerankOptions =
    reranking.mode === "fixed-template"
      ? {
          mode: "fixed-template",
          context: { currentQuery: request.query },
          scorer: reranking.scorer,
        }
      : {
          mode: "customized",
          context: {
            currentQuery: request.query,
            originalQuestion: reranking.originalQuestion,
            searchReason: reranking.searchReason,
          },
          generator: reranking.generator,
          scorer: reranking.scorer,
        };
  const ranked = await rerankCandidates(candidates, options, signal);
  const hitByDocid = new Map(hits.map((hit) => [hit.docid, hit]));
  return {
    ...response,
    hits: ranked.candidates.slice(0, request.limit).map((candidate) => ({
      ...hitByDocid.get(candidate.docid)!,
      rerankScore: candidate.rerankScore,
    })),
    // The backend's nextOffset is not a continuation of this reranked top-k list.
    hasMore: false,
    nextOffset: undefined,
    reranking: {
      criterion: ranked.criterion,
      candidateDocids: candidates.map((candidate) => candidate.docid),
      ranking: ranked.candidates.map(({ docid, retrievalScore, rerankScore }) => ({
        docid,
        retrievalScore,
        rerankScore,
      })),
    },
  };
}
