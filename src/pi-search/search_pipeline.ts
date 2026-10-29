import type { PiSearchBackend } from "./searcher/contract/interface";
import type {
  SearchBackendSearchRequest,
  SearchBackendSearchResponse,
} from "./searcher/contract/types";

// Shared insertion point for search-time processing. For now, preserve backend behavior.
export async function executeSearchPipeline(
  backend: PiSearchBackend,
  request: SearchBackendSearchRequest,
  signal?: AbortSignal,
): Promise<SearchBackendSearchResponse> {
  return backend.search(request, signal);
}
