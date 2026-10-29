import type { PiSearchBackend } from "../searcher/contract/interface";
import type { SearchBackendSearchHit } from "../searcher/contract/types";
import { deduplicateCandidates } from "./candidates";
import type { RerankCandidate } from "./types";

const MAX_DOCUMENT_CHARS = 4000;

/** Read each unique document once and cap the returned text; no pagination or snippet fallback. */
export async function prepareRerankCandidates(
  backend: Pick<PiSearchBackend, "readDocument">,
  hits: readonly SearchBackendSearchHit[],
  signal?: AbortSignal,
): Promise<RerankCandidate[]> {
  signal?.throwIfAborted();
  const candidates: RerankCandidate[] = [];
  for (const hit of deduplicateCandidates(hits)) {
    signal?.throwIfAborted();
    const document = await backend.readDocument({ docid: hit.docid }, signal);
    signal?.throwIfAborted();
    if (!document.found)
      throw new Error(`Cannot prepare rerank candidate: document ${hit.docid} not found.`);
    if (document.docid !== hit.docid)
      throw new Error(`Unexpected document returned for ${hit.docid}.`);
    candidates.push({
      docid: hit.docid,
      title: document.title ?? hit.title ?? undefined,
      text: document.text.slice(0, MAX_DOCUMENT_CHARS),
      retrievalScore: hit.score,
    });
  }
  return candidates;
}
