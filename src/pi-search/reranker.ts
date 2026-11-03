import { isDirectTwoToolInterface, type PiSearchToolInterface } from "./tool_interface";

export type DocumentReranker = (query: string, document: string, signal?: AbortSignal) => Promise<number>;

export const RERANK_INSTRUCTION =
  "Given a web search query, retrieve relevant passages that answer the query";

export function isSearchRerankEnabled(toolInterface: PiSearchToolInterface, env = process.env): boolean {
  return isDirectTwoToolInterface(toolInterface) && /^(1|true)$/i.test(env.PI_SEARCH_RERANK_ENABLED?.trim() ?? "");
}

export function createDocumentReranker(env = process.env, request: typeof fetch = fetch): DocumentReranker {
  const baseUrl = env.PI_SEARCH_RERANK_BASE_URL?.trim();
  const model = env.PI_SEARCH_RERANK_MODEL?.trim();
  if (!baseUrl || !model) throw new Error("Reranking requires PI_SEARCH_RERANK_BASE_URL and PI_SEARCH_RERANK_MODEL.");
  const url = `${baseUrl.replace(/\/$/, "").replace(/\/v1$/, "")}/score`;
  return async (query, document, signal) => {
    const response = await request(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model,
        queries: query,
        documents: [document],
        instruction: RERANK_INSTRUCTION,
        max_tokens_per_doc: 4096,
        use_activation: true,
      }),
      signal,
    });
    if (!response.ok) throw new Error(`Reranker /score failed: HTTP ${response.status}.`);
    const payload = (await response.json()) as { data?: Array<{ index?: number; score?: number }> };
    const row = payload?.data?.[0];
    if (payload?.data?.length !== 1 || row?.index !== 0 ||
      typeof row.score !== "number" || !Number.isFinite(row.score) || row.score < 0 || row.score > 1)
      throw new Error("Reranker returned an invalid score.");
    return row.score;
  };
}
