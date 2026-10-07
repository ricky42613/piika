import {
  filterQuestionSource,
  filterThreshold,
  filterPredictor,
  type FilterQuestionSource,
  type FilterPredictor,
} from "./filter_config";
import { isDirectTwoToolInterface, type PiSearchToolInterface } from "./tool_interface";
import { resolveFilterBackend, type SearchFilterBackend } from "./filter_backends";

export const JEV_GENERAL_PROMPT =
  "Does the document contain information that directly helps answer the query?";

export const FILTER_GUIDANCE =
  "For every search, supply filter_question: a self-contained yes/no question about evidence needed for your current reasoning step. Include relevant entities and context because the filter sees only this question and one document, not your conversation. Ask about a concrete clue or missing link, not generic relevance to the original question; do not require one document to solve every hop. Only documents with P(Yes) > P(No) survive, in their original order. The classifier reads only the first 512 document tokens, so a rejection can miss evidence later in a document. hits is the candidate count before filtering; zero survivors does not prove the answer does not exist. Verify surviving evidence with read_document.";

export function searchFilterGuidance(env = process.env): string {
  const predictor = filterPredictor(env);
  const source = filterQuestionSource(env);
  const threshold = filterThreshold(env);
  if (predictor === "jev") {
    return `Search candidates are assessed with the fixed question: "${JEV_GENERAL_PROMPT}" The tool filters BM25 top REREANK_DEPTH candidates and keeps only documents with true probability > false probability, preserving original BM25 order and returning at most hits survivors. No relevance question or criteria arguments are needed.`;
  }
  if (source === "agent") return FILTER_GUIDANCE.replace("P(Yes) > P(No)", `P(Yes) > ${threshold}`);
  return (
    (source === "general"
      ? "Search candidates are automatically filtered by relevance to the search query."
      : "A filter question is generated automatically from the original question and current search query; candidates are scored against it.") +
    ` Only candidates with score > ${threshold} survive, in their original order. The scorer reads the first 512 document tokens. hits is the candidate count before filtering; zero survivors does not prove no evidence exists. Verify promising documents with read_document.`
  );
}

export function isSearchFilterEnabled(
  toolInterface: PiSearchToolInterface,
  env = process.env,
): boolean {
  return (
    isDirectTwoToolInterface(toolInterface) &&
    /^(1|true)$/i.test(env.PI_SEARCH_FILTER_ENABLED?.trim() ?? "")
  );
}

export type FilterScore = {
  // Logs of the normalized class probabilities, not raw vocabulary token logprobs.
  // null represents log(0), which cannot be represented as a finite JSON number.
  yesLogprob: number | null;
  noLogprob: number | null;
  yesProbability: number;
  noProbability: number;
  keep: boolean;
  probabilitySemantics: "normalized_yes_no_tokens";
  backend?: SearchFilterBackend;
  predictor?: FilterPredictor;
  questionSource?: FilterQuestionSource;
  threshold?: number;
  promptVersion?: string;
  scoringApi?: "score" | "generative_scoring" | "systemone" | "vllm_prompt_logprobs";
  promptTokensSource?: "server_usage" | "constructed_token_ids";
  model?: string;
  promptTokens?: number;
  documentTokens?: number;
  documentTruncated?: boolean;
  scoringRequests?: number;
};
export type DocumentFilter = (
  question: string,
  document: string,
  signal?: AbortSignal,
) => Promise<FilterScore>;

function logsumexp(values: readonly number[]): number {
  const max = Math.max(...values);
  return max + Math.log(values.reduce((sum, value) => sum + Math.exp(value - max), 0));
}

export function filterScoreFromYesProbability(yes: number, threshold = 0.5): FilterScore {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1)
    throw new Error("Invalid filter threshold.");
  if (!Number.isFinite(yes) || yes < 0 || yes > 1) {
    throw new Error("Search filter requires a Yes probability in [0, 1].");
  }
  const no = 1 - yes;
  return {
    yesLogprob: yes > 0 ? Math.log(yes) : null,
    noLogprob: no > 0 ? Math.log(no) : null,
    yesProbability: yes,
    noProbability: no,
    keep: yes > threshold,
    threshold,
    probabilitySemantics: "normalized_yes_no_tokens",
  };
}

const MAX_PROMPT_TOKENS = 8191; // Reserve one position in the 8192-token model context.
const MAX_DOCUMENT_TOKENS = 512;

/** Shared scoring transport/policy with model-specific prompt and label adapters. */
export function createYesNoFilter(
  env = process.env,
  request: typeof fetch = fetch,
): DocumentFilter {
  const profile = resolveFilterBackend(env);
  const { baseUrl, model, apiKey } = profile;
  const threshold = filterThreshold(env);
  const predictor = filterPredictor(env);
  const questionSource = predictor === "jev" ? "general" : filterQuestionSource(env);
  let labels: { yes: number[]; all: number[]; suffix: number[] } | undefined;

  async function post(route: string, body: unknown, signal?: AbortSignal): Promise<unknown> {
    const timeout = AbortSignal.timeout(180_000);
    const response = await request(`${baseUrl}${route}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify(body),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    if (!response.ok)
      throw new Error(
        `Search filter HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`,
      );
    return response.json();
  }

  async function tokenize(prompt: string, signal?: AbortSignal): Promise<number[]> {
    const data = (await post(
      "/tokenize",
      { model, prompt, add_special_tokens: false },
      signal,
    )) as { tokens?: unknown } | null;
    if (
      !Array.isArray(data?.tokens) ||
      !data.tokens.every(
        (id: unknown) => typeof id === "number" && Number.isSafeInteger(id) && id >= 0,
      )
    ) {
      throw new Error("Search filter received invalid tokenizer token IDs.");
    }
    return data.tokens as number[];
  }

  async function variants(texts: string[], signal?: AbortSignal): Promise<number[]> {
    const ids = new Set<number>();
    for (const text of texts) {
      const tokens = await tokenize(text, signal);
      if (tokens.length === 1) ids.add(tokens[0]!);
      else throw new Error(`Search filter ${profile.backend} requires single-token ${text}.`);
    }
    if (!ids.size)
      throw new Error(`Search filter requires at least one single-token ${texts[0]} variant.`);
    return [...ids].sort((a, b) => a - b);
  }

  return async (question, document, signal) => {
    signal?.throwIfAborted();
    if (!question.trim()) throw new Error("Search filter requires a non-empty filter question.");
    if (profile.scoringApi === "systemone") {
      const payload = (await post(
        "/v1/systemone",
        {
          state: { query: question, candidate_passage: document },
          model,
          questions: {
            relevant: {
              type: "noul",
              instructions: JEV_GENERAL_PROMPT,
            },
          },
        },
        signal,
      )) as {
        model?: string;
        answers?: { relevant?: { noul?: number; probabilities?: Record<string, number> } };
        usage?: { input_tokens?: number; output_tokens?: number };
      } | null;
      const answer = payload?.answers?.relevant;
      let trueProbability = typeof answer?.noul === "number" ? answer.noul : undefined;
      const probs = answer?.probabilities;
      if (trueProbability === undefined && probs) {
        trueProbability = probs.true ?? probs.True ?? probs.relevant ?? probs["1"];
      }
      const falseProbability = probs
        ? (probs.false ?? probs.False ?? probs.irrelevant ?? probs["0"])
        : trueProbability === undefined
          ? undefined
          : 1 - trueProbability;
      if (
        typeof trueProbability !== "number" ||
        !Number.isFinite(trueProbability) ||
        trueProbability < 0 ||
        trueProbability > 1 ||
        typeof falseProbability !== "number" ||
        !Number.isFinite(falseProbability) ||
        falseProbability < 0 ||
        falseProbability > 1
      )
        throw new Error("Search filter received invalid JEV noul probabilities.");
      const promptTokens = payload?.usage?.input_tokens;
      if (promptTokens !== undefined && (!Number.isSafeInteger(promptTokens) || promptTokens < 0))
        throw new Error("Search filter received invalid JEV input token usage.");
      return {
        yesLogprob: trueProbability > 0 ? Math.log(trueProbability) : null,
        noLogprob: falseProbability > 0 ? Math.log(falseProbability) : null,
        yesProbability: trueProbability,
        noProbability: falseProbability,
        keep: trueProbability > falseProbability,
        threshold: falseProbability,
        predictor,
        questionSource,
        backend: profile.backend,
        promptVersion: profile.promptVersion,
        scoringApi: profile.scoringApi,
        probabilitySemantics: profile.probabilitySemantics,
        model: payload?.model ?? model,
        ...(promptTokens !== undefined
          ? { promptTokens, promptTokensSource: "server_usage" as const }
          : {}),
        documentTokens: undefined,
        documentTruncated: undefined,
        scoringRequests: 1,
      };
    }
    if (!labels) {
      const yes =
        profile.scoringApi === "score" || profile.scoringApi === "vllm_prompt_logprobs"
          ? []
          : await variants(profile.yesLabels, signal);
      const no =
        profile.scoringApi === "score" || profile.scoringApi === "vllm_prompt_logprobs"
          ? []
          : await variants(profile.noLabels, signal);
      if (yes.some((id) => no.includes(id)))
        throw new Error("Search filter Yes/No token variants overlap.");
      labels = { yes, all: [...yes, ...no], suffix: await tokenize(profile.suffix, signal) };
    }
    // Preserve complete instructions/question/suffix, truncate only the document.
    // Qwen uses its chat template; only document content may be truncated.
    const prefix = await tokenize(profile.prefix(question), signal);
    const budget = Math.min(
      MAX_DOCUMENT_TOKENS,
      MAX_PROMPT_TOKENS - prefix.length - labels.suffix.length,
    );
    if (budget < 1)
      throw new Error(
        "Search filter question is too long to preserve question and answer instructions within 8191 tokens.",
      );
    const docIds = await tokenize(document, signal);
    const retainedDocIds = docIds.slice(0, budget);
    if (profile.scoringApi === "score") {
      // The pooling server owns the chat template and lowercase yes/no classifier.
      // Ask it to truncate only the document, never the question or combined prompt.
      const payload = (await post(
        "/score",
        {
          model,
          queries: question,
          documents: [document],
          instruction: profile.instruction,
          max_tokens_per_query: 0,
          max_tokens_per_doc: budget,
          truncate_prompt_tokens: null,
          truncation_side: "right",
          use_activation: true,
        },
        signal,
      )) as {
        data?: Array<{ index?: number; score?: number }>;
        usage?: { prompt_tokens?: number };
      } | null;
      const row = payload?.data?.[0];
      if (
        payload?.data?.length !== 1 ||
        row?.index !== 0 ||
        typeof row.score !== "number" ||
        !Number.isFinite(row.score) ||
        row.score < 0 ||
        row.score > 1
      ) {
        throw new Error(
          "Search filter received an invalid or incomplete score response (expected an activated probability in [0, 1]).",
        );
      }
      const promptTokens = payload?.usage?.prompt_tokens;
      if (promptTokens !== undefined && (!Number.isSafeInteger(promptTokens) || promptTokens < 0))
        throw new Error("Search filter received invalid score prompt token usage.");
      return {
        ...filterScoreFromYesProbability(row.score, threshold),
        predictor,
        questionSource,
        backend: profile.backend,
        promptVersion: profile.promptVersion,
        scoringApi: profile.scoringApi,
        probabilitySemantics: profile.probabilitySemantics,
        model,
        ...(promptTokens !== undefined
          ? { promptTokens, promptTokensSource: "server_usage" as const }
          : {}),
        documentTokens: retainedDocIds.length,
        documentTruncated: retainedDocIds.length < docIds.length,
        scoringRequests: 1,
      };
    }
    if (profile.scoringApi === "vllm_prompt_logprobs") {
      const promptText = profile.prefix(question) + document.slice(0, 16000) + profile.suffix;
      async function forcedLabelLogprob(label: string): Promise<number> {
        const payload = (await post(
          "/v1/completions",
          {
            model,
            prompt: [promptText + label],
            max_tokens: 1,
            temperature: 0,
            prompt_logprobs: 1,
          },
          signal,
        )) as {
          choices?: Array<{
            index?: number;
            prompt_logprobs?: Array<Record<string, { decoded_token?: string; logprob?: number }> | null>;
          }>;
        } | null;
        const top = payload?.choices?.[0]?.prompt_logprobs?.at(-1);
        if (!top) throw new Error("Search filter received invalid vLLM prompt_logprobs response.");
        for (const entry of Object.values(top)) {
          if (entry?.decoded_token === label && typeof entry.logprob === "number") {
            return entry.logprob;
          }
        }
        throw new Error(`Search filter vLLM response did not include forced token ${label}.`);
      }
      const yesLogprob = await forcedLabelLogprob(profile.yesLabels[0]!);
      const noLogprob = await forcedLabelLogprob(profile.noLabels[0]!);
      const z = logsumexp([yesLogprob, noLogprob]);
      const yesProbability = Math.exp(yesLogprob - z);
      const noProbability = Math.exp(noLogprob - z);
      return {
        yesLogprob,
        noLogprob,
        yesProbability,
        noProbability,
        keep: yesProbability > noProbability,
        threshold: noProbability,
        predictor,
        questionSource,
        backend: profile.backend,
        promptVersion: profile.promptVersion,
        scoringApi: profile.scoringApi,
        probabilitySemantics: profile.probabilitySemantics,
        model,
        promptTokens: prefix.length + retainedDocIds.length + labels.suffix.length,
        promptTokensSource: "constructed_token_ids",
        documentTokens: retainedDocIds.length,
        documentTruncated: retainedDocIds.length < docIds.length,
        scoringRequests: 2,
      };
    }
    const prompt = [...prefix, ...retainedDocIds, ...labels.suffix];
    const payload = (await post(
      "/generative_scoring",
      {
        model,
        query: [],
        items: [prompt],
        label_token_ids: labels.all,
        apply_softmax: true,
        add_special_tokens: false,
      },
      signal,
    )) as { data?: Array<{ index?: number; score?: number }> } | null;
    const row = payload?.data?.[0];
    if (
      payload?.data?.length !== 1 ||
      row?.index !== 0 ||
      typeof row.score !== "number" ||
      !Number.isFinite(row.score) ||
      row.score < 0 ||
      row.score > 1
    ) {
      throw new Error("Search filter received an invalid or incomplete generative scoring response.");
    }
    return {
      ...filterScoreFromYesProbability(row.score, threshold),
      predictor,
      questionSource,
      backend: profile.backend,
      promptVersion: profile.promptVersion,
      scoringApi: profile.scoringApi,
      promptTokensSource: "constructed_token_ids",
      probabilitySemantics: profile.probabilitySemantics,
      model,
      promptTokens: prompt.length,
      documentTokens: retainedDocIds.length,
      documentTruncated: retainedDocIds.length < docIds.length,
      scoringRequests: 1,
    };
  };
}
