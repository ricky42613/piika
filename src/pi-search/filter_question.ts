// Qorrect rollout-4 policy prompt and generation settings:
// https://huggingface.co/ricky42613/qorrect-rollout4 (prompts.py, policy.py, README.md).
// This model generates questions only; document classification uses a separate predictor.
export type FilterQuestionGenerator = (
  currentQuery: string,
  currentReason: string,
  signal?: AbortSignal,
  seenDocs?: readonly string[],
) => Promise<string>;

export function qorrectSeenDocsEnabled(env = process.env): boolean {
  const value = env.PI_SEARCH_FILTER_GENERATOR_INCLUDE_SEEN_DOCS?.trim().toLowerCase();
  if (!value || value === "0" || value === "false") return false;
  if (value === "1" || value === "true") return true;
  throw new Error("PI_SEARCH_FILTER_GENERATOR_INCLUDE_SEEN_DOCS must be 0, 1, false, or true.");
}

export function buildFilterQuestionPrompt(
  originalQuestion: string,
  currentQuery: string,
  currentReason = "",
  seenDocs: readonly string[] = [],
  enableSeenDocs = false,
): string {
  const seenSection = enableSeenDocs && seenDocs.length
    ? `Previously seen documents:\n${seenDocs.join("\n\n")}\nTreat then as background background knowledge you already have.\n\n`
    : "";
  // Match the rollout-4 prompts.py builder exactly, including its seen-docs wording.
  return `You are an agent performing multi-step search.

Original question:
${originalQuestion}

Reason for current search:
${currentReason}
Current search query:
${currentQuery}

${seenSection}Write ONE complete yes/no question to filter individual retrieved documents.

Choose the scope of the question based on the CURRENT search state and information need. The question may be broad when exploring possible leads or narrower when checking a particular fact, candidate, or relationship.

Rules:
- Output only ONE complete yes/no question, with no answer, explanation, or labels.
- The question must be answerable from ONE document.
- Do not mention document IDs or ranks.
- Do not encode answers for individual documents.

Filter question:`;
}

// Qwen3.5 chat template with enable_thinking=False and add_generation_prompt=True.
export function qwenChatPrompt(userPrompt: string): string {
  return `<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

export function createFilterQuestionGenerator(
  env = process.env,
  request: typeof fetch = fetch,
): FilterQuestionGenerator {
  // Generator settings never inherit the predictor endpoint/model/API key.
  const baseUrl = (env.PI_SEARCH_FILTER_GENERATOR_BASE_URL?.trim() || "http://127.0.0.1:18080/v1")
    .replace(/\/$/, "")
    .replace(/\/v1$/, "");
  const model = env.PI_SEARCH_FILTER_GENERATOR_MODEL?.trim() || "ricky42613/qorrect-rollout4";
  const apiKey = env.PI_SEARCH_FILTER_GENERATOR_API_KEY;
  const originalQuestion = env.PI_SEARCH_ORIGINAL_QUESTION?.trim();
  const maxTokens = Number(env.PI_SEARCH_FILTER_GENERATOR_MAX_TOKENS?.trim() || "64");
  const temperature = Number(env.PI_SEARCH_FILTER_GENERATOR_TEMPERATURE?.trim() || "1");
  const includeSeenDocs = qorrectSeenDocsEnabled(env);
  if (!Number.isInteger(maxTokens) || maxTokens < 1)
    throw new Error("PI_SEARCH_FILTER_GENERATOR_MAX_TOKENS must be a positive integer.");
  if (!Number.isFinite(temperature) || temperature <= 0)
    throw new Error("PI_SEARCH_FILTER_GENERATOR_TEMPERATURE must be a positive number.");
  if (!originalQuestion)
    throw new Error(
      "Qorrect question generation requires PI_SEARCH_ORIGINAL_QUESTION (set automatically by the benchmark runner).",
    );
  return async (currentQuery, currentReason, signal, seenDocs = []) => {
    signal?.throwIfAborted();
    if (!currentQuery.trim()) throw new Error("Question generation requires a non-empty search query.");
    const timeout = AbortSignal.timeout(180_000);
    const prompt = qwenChatPrompt(buildFilterQuestionPrompt(
      originalQuestion, currentQuery, currentReason, seenDocs, includeSeenDocs,
    ));
    const response = await request(`${baseUrl}/v1/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify({
        model,
        prompt: [prompt],
        max_tokens: maxTokens,
        temperature,
        top_p: 1.0,
        top_k: 0,
      }),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    if (!response.ok)
      throw new Error(
        `Filter question generation HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`,
      );
    const data = (await response.json()) as {
      choices?: Array<{ index?: number; finish_reason?: string; text?: unknown }>;
    };
    const choice = [...(data?.choices ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0))[0];
    const text = choice?.text;
    if (typeof text !== "string" || !text.trim()) {
      throw new Error("Qorrect returned an empty or invalid filter question.");
    }
    return text.trim();
  };
}
