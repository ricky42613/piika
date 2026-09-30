export type FilterQuestionGenerator = (context: string, signal?: AbortSignal) => Promise<string>;

export function buildFilterQuestionPrompt(originalQuestion: string, context: string): string {
  return `You are an agent performing multi-step search.

Original question:
${originalQuestion}

Current search context:
${context}

Write ONE concise yes/no question that can be used to filter the
retrieved documents.

The question should identify whether a document contains the specific
information currently needed by the agent.

Rules:
- Output only the question.
- It must be answerable with Yes or No.
- Do not mention document IDs or ranks.
- Do not encode answers for individual documents.

Filter question:`;
}

export function createFilterQuestionGenerator(
  env = process.env,
  request: typeof fetch = fetch,
): FilterQuestionGenerator {
  // Generator settings never inherit the predictor endpoint/model/API key.
  const baseUrl = (env.PI_SEARCH_FILTER_GENERATOR_BASE_URL?.trim() || "http://127.0.0.1:8150/v1").replace(/\/$/, "").replace(/\/v1$/, "");
  const model = env.PI_SEARCH_FILTER_GENERATOR_MODEL?.trim() || "reranq-filter";
  const apiKey = env.PI_SEARCH_FILTER_GENERATOR_API_KEY;
  const originalQuestion = env.PI_SEARCH_ORIGINAL_QUESTION?.trim();
  if (!originalQuestion) throw new Error("Qorrect question generation requires PI_SEARCH_ORIGINAL_QUESTION (set automatically by the benchmark runner).");
  return async (context, signal) => {
    signal?.throwIfAborted();
    if (!context.trim()) throw new Error("Question generation requires a non-empty search query.");
    const timeout = AbortSignal.timeout(180_000);
    const response = await request(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: buildFilterQuestionPrompt(originalQuestion, context) }],
        temperature: 0,
        max_tokens: 64,
        chat_template_kwargs: { enable_thinking: false },
      }),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    if (!response.ok) throw new Error(`Filter question generation HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
    const data = await response.json() as { choices?: Array<{ finish_reason?: string; message?: { content?: unknown } }> };
    const choice = data?.choices?.[0];
    const text = choice?.message?.content;
    if (choice?.finish_reason !== "stop" || typeof text !== "string" || !text.trim()) {
      throw new Error("Qorrect returned an empty, invalid or incomplete filter question.");
    }
    return text.trim();
  };
}
