# `pi-search`

`pi-search` is the package-owned search extension surface inside this repository.

It is designed to be a **backend-agnostic, extraction-ready extension boundary**:

- it owns the agent-facing retrieval tools
- it owns the extension contract and validation rules
- it owns extension-local runtime behavior such as cached search sessions, pagination, spill handling, and prompt policy
- it does **not** own repo-local benchmark orchestration or provider-specific transport/process wiring

If `pi-search` is ever extracted into its own package, this directory is intended to move with minimal churn.

For the deeper repo-level ownership contract, see:

- `../../docs/pi-search-contract.md`

For the current repo-local Anserini integration seam, see:

- `../../docs/bm25-extension-interface.md`

For the Pyserini REST backend, see:

- `../../docs/pyserini-rest-search-provider.md`

## What lives here

`src/pi-search/` owns the standalone extension contract and package-local behavior.

That includes:

- extension registration
- extension config parsing
- tool argument / result schemas
- runtime validation and protocol error shaping
- searcher contract definitions
- adapter normalization into the shared searcher contract
- backend lifecycle/cache behavior inside the extension runtime
- search session state and `search_id` pagination
- read/search spill behavior
- prompt shaping and time-budget steering

## What does not live here

The following remain outside `src/pi-search/` on purpose:

- benchmark orchestration
- benchmark artifact writing and evaluation
- operator/TUI code
- repo-local extension wrappers
- repo-local provider transport/process/client construction

Current repo-local Anserini provider code lives under:

- `src/search-providers/anserini/`

That provider layer is injected into package-owned `pi-search` from outside, rather than being imported directly by package-owned modules.

## Main entrypoints

### `extension.ts`

Package-owned extension registration and composition.

Main export:

- `registerPiSearchExtension(...)`

### Optional yes/no search filter (two-tool interface only)

Set `PI_SEARCH_FILTER_ENABLED=1` (or `true`) before launching the agent. Default is off;
`pi-serini-3tool` is unchanged even when this variable is set. Restart/reload the extension
when changing the flag. Enabled `search` requires `filter_question`, a self-contained yes/no
question tailored to the current reasoning step, in addition to `reason`, `query`, and optional `hits`.

```json
{"reason":"Find evidence for the candidate's education","query":"Ada Lovelace mathematics education","hits":10,"filter_question":"Does this document describe Ada Lovelace's mathematics education or tutors?"}
```

`hits` is the number of candidates **before** filtering (no automatic refill). The tool fetches
each candidate document, following pagination, then classifies its **first 512 tokens** using
the selected model (`reranq` by default). It keeps a document only if `P(Yes) > P(No)`. Ties are removed
(intentionally stricter than the reference client's `>= 0.5`); surviving order and retrieval
scores are unchanged. The agent still verifies evidence with `read_document`. Rejection can
miss evidence beyond the document prefix; zero survivors is not proof of absence.

Select the model adapter with `PI_SEARCH_FILTER_BACKEND=reranq|qwen3-reranker`.
Omitting it preserves the existing Reranq behavior. Unknown values fail at initialization
when filtering is enabled; disabled filtering and the three-tool interface ignore this setting.
Restart/reload the extension after changing backend settings.

| Backend | Default model | Endpoint | Labels |
|---|---|---|---|
| `reranq` (default) | `reranq-filter` | `http://127.0.0.1:8150/v1` | `Yes`, ` Yes`, `No`, ` No` single-token variants |
| `qwen3-reranker` | `Qwen/Qwen3-Reranker-4B` | Explicit `PI_SEARCH_FILTER_BASE_URL` required | Exactly `yes`, `no` |

Override with `PI_SEARCH_FILTER_BASE_URL`, `PI_SEARCH_FILTER_MODEL`, and optionally
`PI_SEARCH_FILTER_API_KEY` (`RERANQ_API_KEY` fallback applies **only** to the Reranq backend).
The base URL may include `/v1` or be the server root. Clear stale endpoint/model overrides
when switching models. `PI_SEARCH_FILTER_MODEL` must match the server's served model name.
Historical run scripts/snapshots are not modified.

**Serving requirements:**

- Reranq uses `/tokenize` + `/generative_scoring` (a nonstandard endpoint supported by the
  existing Reranq vLLM 0.22.1 service). It accepts complete prompt token IDs and explicit
  `label_token_ids` with `apply_softmax=true`, returning the probability of the **first**
  label as `data[0].score`.
- Qwen defaults to `/tokenize` + `/score`, compatible with the pooling service at
  `http://127.0.0.1:8152`, served model name `qwen3-reranker-4b`. The server must use
  Qwen's reranker chat template (honoring the request's `instruction`) and the lowercase
  `no`/`yes` classifier. Requests set `use_activation=true`; scores must be normalized
  probabilities, **not raw logits**. The service must honor `max_tokens_per_doc`,
  `max_tokens_per_query=0`, and `truncate_prompt_tokens=null`: truncate only the document,
  preserve the whole question/template, and error on context overflow.
- Optionally, Qwen can use the token-ID protocol via
  `PI_SEARCH_FILTER_API=generative_scoring`; otherwise its default is `score`.
  Reranq supports only `generative_scoring`. No automatic protocol fallback occurs.

Use at least 8192 tokens of server context. `/rerank` is deliberately unused because
Piika keeps original retrieval order. A generic OpenAI chat endpoint is not interchangeable.
The Hugging Face model URL is not an inference URL; Piika does not download/deploy weights
or allocate GPUs. No fallback to chat completions or unfiltered retrieval is performed.

Reranq classification matches `reranq/prompts.py` and `serving/client.py`:

```text
Filter question:
{filter_question}

Document:
{document first 512 tokens}

Answer the filter question using only this document.
Answer exactly Yes or No.

Answer:
```

Prefix, document and suffix are tokenized separately via the model server's `/tokenize`
(`add_special_tokens=false`); Reranq uses no chat template. Document budget is
`min(512, 8191 - prefixTokens - suffixTokens)`, reserving one position in the 8192-token
context. The question and instructions are never truncated; an oversized question errors.
No local tokenizer, Python dependency, or model weights are needed in Piika.

For Reranq, the client obtains the single-token variants among `Yes`, ` Yes`, `No`, and ` No`, deduplicates
them within each class, and sends complete prompt IDs to `/generative_scoring`. It queries
each Yes variant as the first label with the same normalization set and `apply_softmax=true`,
then sums their scores. `noProbability = 1 - yesProbability`.
These are **normalized Yes/No class probabilities**, not the old full-vocabulary raw token
probabilities. Tool details explicitly record backend, prompt version, `probabilitySemantics`, model, prompt tokens,
retained document tokens, truncation, scoring request count and decisions. The existing
`yesLogprob`/`noLogprob` fields now mean logs of class probabilities (`null` for log(0)).
Token counts describe the scoring prompt, not provider-reported/billed usage.

Documents are evaluated sequentially; each HTTP request has a 180-second timeout and honors
cancellation. Missing documents, invalid scores, tokenization errors, context overflow, and
HTTP errors fail the tool call rather than silently bypassing filtering. Full documents are
sent to the tokenizer endpoint for counting. With `generative_scoring`, only the retained
prefix token IDs are sent for scoring. With `score`, the full document text is sent and the
server applies the requested document-only token budget before constructing the prompt.

#### Qwen3-Reranker filter ablation

The Qwen adapter follows the [official Transformers template](https://huggingface.co/Qwen/Qwen3-Reranker-4B),
including its system message and non-thinking assistant suffix. It inserts:

- `<Instruct>`: judge whether the document provides sufficient evidence to answer **Yes**;
  support for No, mere topic overlap, and insufficient evidence do not qualify.
- `<Query>`: the agent's original `filter_question`, not its BM25 query string.
- `<Document>`: first 512 document tokens (or less if context space is exhausted).

Qwen uses lowercase `yes` and `no`, no whitespace variants. One scoring request returns
the activated Yes probability (`softmax([logit_yes, logit_no])[0]`); details record
`probabilitySemantics="normalized_yes_no_tokens"`, `scoringApi`, and
`promptVersion="qwen3-affirmative-evidence-v1"`.

With `/score`, Piika sends raw `queries`, `documents`, and `instruction` and the server
applies the chat template exactly once. `/tokenize` is used for document accounting and
an explicit question/template context-budget check. The server must match the documented
Qwen template. `promptTokens` comes from the response's `usage.prompt_tokens`, marked
`promptTokensSource="server_usage"`; it is omitted if usage is absent, never fabricated.
Document counts describe the original token-prefix budget; server decode/re-encode may
change token boundaries. Context overflow remains a tool failure, not silent truncation.

With `/generative_scoring`, Piika builds the prompt IDs and the server must not apply a
second chat template. `promptTokensSource="constructed_token_ids"` marks the derived count,
not provider-reported usage. Both paths preserve the question and assistant suffix.

This is still **filtering, not reranking**: both backends use the strict `score > 0.5` rule,
keep original retrieval order/scores, and never refill. No threshold/top-k policy changes
are introduced. Equal thresholds do not imply equally calibrated scores or retention rates.
512 tokens can cover different text under different tokenizers; for an offline replay
comparison, fix the same question, candidate set, and document text prefix. The replay
pipeline itself is not part of this backend switch.

Example DL19 exact-30 run with the deployed Qwen pooling service:

```bash
PI_SEARCH_FILTER_ENABLED=1 \
PI_SEARCH_FILTER_BACKEND=qwen3-reranker \
PI_SEARCH_FILTER_BASE_URL=http://127.0.0.1:8152 \
PI_SEARCH_FILTER_MODEL=qwen3-reranker-4b \
PI_SEARCH_FILTER_API=score \
PYSERINI_REST_BASE_URL=http://localhost:8081 \
PYSERINI_REST_INDEX=msmarco-v1-passage \
npm run run:benchmark:query-set:sharded-shared-bm25 -- \
  --benchmark msmarco-v1-passage --query-set dl19 \
  --model qwen-local/qwen38-27b-nvfp4 --thinking medium \
  --shard-count 2 --timeout-seconds 900 \
  --tool-interface pyserini-rest-2tool \
  --output-mode ranked_list --ranked-list-depth 30 --ranked-list-count 30 \
  --output-root "runs/dl19_qwen38_qwen3reranker_exact30_$(date +%Y%m%d_%H%M%S)" \
  --no-auto-evaluate-on-merge
```

Use `--dry-run` first (validates the launch plan, **not** the remote scoring service).
For the Reranq arm, change backend/base URL/model together to
`reranq`, `http://127.0.0.1:8150/v1`, `reranq-filter`.
For the no-filter arm, set `PI_SEARCH_FILTER_ENABLED=0`.
Keep agent, query set, document budget, hits policy, timeout, and exact output count fixed.
Record endpoint/model/backend settings with the run; successful per-document scores also
persist the selected backend/model/prompt version for auditing.

By default this file registers the direct two-tool interface:

- `search`
- `read_document`

The `pi-serini-3tool` interface is an explicit opt-in and additionally registers
`read_search_results` for cached ranking pagination. Prompt generation receives the resolved
interface so it never instructs the agent to call a tool that is not registered.

It accepts injected backend creation rather than constructing repo-local transport/process details itself.

### `config.ts`

Owns the `PI_SEARCH_EXTENSION_CONFIG` contract.

Important exports include:

- `PiSearchExtensionConfigSchema`
- `parsePiSearchExtensionConfig(...)`
- `resolvePiSearchExtensionConfigFromEnv(...)`
- `buildAnseriniBm25TcpExtensionConfig(...)`
- `buildAnseriniBm25StdioExtensionConfig(...)`
- `buildHttpJsonExtensionConfig(...)`
- `buildMockExtensionConfig(...)`

The extension now requires explicit config from its caller.

### `agent_prompt.ts`

Owns the reusable `pi-search` retrieval workflow prompt template.

This is package-owned prompt content for the `pi-search` retrieval flow, not generic repo runtime infrastructure.

## Directory map

### `protocol/`

Owns extension-facing schemas, TypeBox validation wiring, parsers, protocol error types, and typed detail extractors.

Key responsibilities:

- tool parameter schemas
- tool result / payload schemas
- boundary parsing and validation
- repair-friendly protocol error shaping
- small stable machine-readable metadata for non-prose consumers

### `searcher/contract/`

Owns the backend-agnostic searcher contract.

This is the normalized interface that all supported backends must satisfy.

It defines things such as:

- search requests/responses
- read-document requests/responses
- backend capabilities
- backend-specific validation errors for malformed or invalid responses

### `searcher/adapters/`

Owns package-owned adapters that normalize concrete backends into the shared `PiSearchBackend` contract.

Current adapters:

- `anserini_bm25/`
- `http_json/`
- `mock/`
- `pyserini_rest/`
- `create.ts`

Important boundary rule:

- package-owned adapters may translate backend behavior into the shared contract
- they must not import repo-owned orchestration/evaluation/operator/provider layers

### `searcher/runtime.ts`

Owns backend caching/lifecycle behavior inside the extension runtime.

This is where backend instances are cached and disposed. It is also where caller-injected backend construction hooks plug into package-owned `pi-search`.

### `tool_handlers.ts`

Owns the concrete behavior behind:

- `search`
- `read_search_results`
- `read_document`

This file is responsible for applying the contract and producing repair-friendly extension behavior.

### `search_cache.ts`

Owns cached search session state, page construction, and ranked-result browsing helpers.

### `spill.ts`

Owns truncation-aware spill-file behavior for large search or document reads.

### `prompt_policy.ts`

Owns package-local prompt shaping and time-budget steering behavior.

This currently still reflects the benchmark harness environment the extension runs inside, but it remains package-owned because it is part of the extension’s retrieval workflow policy.

### `tool_types.ts`

Owns shared typed detail structures used by tool handlers and downstream contract consumers.

## Current supported backend kinds

`pi-search` currently supports these backend kinds through its config and shared searcher contract:

- `anserini-bm25`
- `http-json`
- `mock`
- `pyserini-rest`

Important distinction:

- `anserini-bm25` is a supported backend kind in the package contract
- repo-local transport/process construction for the in-repo Anserini path still lives outside this directory under `src/search-providers/anserini/`

## Dependency direction

The intended dependency direction is strict:

- package-owned `src/pi-search/` code may depend on neutral/shared runtime utilities and external libraries
- repo-owned wrappers, orchestration, evaluation, operator code, and provider-integration layers may depend on `src/pi-search/`
- package-owned `src/pi-search/` code must not depend back on repo-owned layers

In practice, `src/pi-search/` should stay free of direct imports from repo-owned areas such as:

- `src/search-providers/`
- `src/orchestration/`
- `src/evaluation/`
- `src/operator/`
- `src/benchmarks/`
- `src/extensions/`

## Guardrails

This boundary is protected by tests.

In particular:

- `tests/pi-search/extension.test.ts`

contains whole-tree regression guards that fail if package-owned `src/pi-search/` starts importing repo-owned layers again.

## Relationship to the repo-local wrapper

The repository still needs a composition point that wires package-owned `pi-search` to repo-local provider construction.

That wrapper lives at:

- `src/extensions/pi_search.ts`

Its job is intentionally narrow:

- import the repo-local backend factory
- inject it into `registerPiSearchExtension(...)`

That keeps the dependency direction honest:

- `pi-search` owns the package boundary
- the repository owns provider composition

## If extraction happens later

The target shape is already visible:

- `src/pi-search/` is the movable package-owned body
- repo-local wrappers and provider code stay outside it

That does **not** mean extraction is required now.

It means this directory should keep behaving like a package boundary now, so a future extraction decision is based on product/release needs rather than cleanup debt.
