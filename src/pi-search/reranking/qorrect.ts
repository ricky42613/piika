import type { CriterionContext, CriterionGenerator } from "./criterion";

export type QorrectGeneratorConfig = {
  /** OpenAI-compatible API base, including /v1 (for example http://localhost:8000/v1). */
  baseUrl: string;
  model: string;
  apiKey?: string;
  maxTokens?: number;
  temperature?: number;
  timeoutMs?: number;
};

export function createQorrectGenerator(
  config: QorrectGeneratorConfig,
  fetchImpl: typeof fetch = fetch,
): CriterionGenerator {
  const endpoint = new URL(`${config.baseUrl.replace(/\/+$/, "")}/chat/completions`);
  if (!["http:", "https:"].includes(endpoint.protocol)) {
    throw new Error("Qorrect baseUrl must use HTTP or HTTPS.");
  }
  if (!config.model.trim()) throw new Error("Qorrect model must be non-empty.");
  const maxTokens = config.maxTokens ?? 64;
  const temperature = config.temperature ?? 0.3;
  const timeoutMs = config.timeoutMs ?? 30_000;
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) {
    throw new Error("Qorrect maxTokens must be a positive integer.");
  }
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    throw new Error("Qorrect temperature must be between 0 and 2.");
  }
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) {
    throw new Error("Qorrect timeoutMs must be a positive integer within the timer range.");
  }

  return {
    async generateQuestion(context, signal) {
      const timeout = AbortSignal.timeout(timeoutMs);
      const response = await fetchImpl(endpoint, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {}),
        },
        body: JSON.stringify({
          model: config.model,
          messages: [{ role: "user", content: buildQorrectPrompt(context) }],
          chat_template_kwargs: { enable_thinking: false },
          max_tokens: maxTokens,
          temperature,
          stream: false,
          n: 1,
        }),
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
      if (!response.ok) {
        // Avoid exposing server response bodies, which may contain prompts or credentials.
        throw new Error(`Qorrect generation failed: HTTP ${response.status}.`);
      }
      const payload: unknown = await response.json();
      if (
        !payload || typeof payload !== "object" || !("choices" in payload) ||
        !Array.isArray(payload.choices) || payload.choices.length !== 1
      ) {
        throw new Error("Invalid Qorrect response: expected exactly one choice.");
      }
      const choice = payload.choices[0];
      if (!choice || choice.finish_reason !== "stop") {
        throw new Error("Qorrect generation did not finish normally (expected finish_reason=stop).");
      }
      const content: unknown = choice.message?.content;
      if (typeof content !== "string" || !content.trim()) {
        throw new Error("Invalid Qorrect response: expected a non-empty question.");
      }
      // Full-output checkpoint: preserve all lines, not just the first question line.
      return content;
    },
  };
}

/**
 * Matches build_question_prompt(..., enable_seen_docs=False) in
 * ricky42613/qorrect-fulloutput717-pplx-flash/prompts.py.
 * This is user-message content, not an already chat-templated model input.
 */
export function buildQorrectPrompt(context: CriterionContext): string {
  return `You are an agent performing multi-step search.

Original question:
${context.originalQuestion}

Reason for current search:
${context.searchReason ?? ""}
Current search query:
${context.currentQuery}

Write ONE complete yes/no question to filter individual retrieved documents.

Choose the scope of the question based on the CURRENT search state and information need. The question may be broad when exploring possible leads or narrower when checking a particular fact, candidate, or relationship.

Rules:
- Output only ONE complete yes/no question, with no answer, explanation, or labels.
- The question must be answerable from ONE document.
- The document scorer sees ONLY your generated question and the document, not the original question, search query, or search reason above. Make the generated question self-contained by stating the relevant topic, entity, or information need explicitly.
- Do not use unresolved references such as "my question", "the user's question", or "the search query" unless you explicitly state the referenced information in the generated question. A broad but self-contained relevance question is allowed.
- Do not mention document IDs or ranks.
- Do not encode answers for individual documents.

Filter question:`;
}
