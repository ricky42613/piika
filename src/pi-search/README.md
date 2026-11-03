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

### Optional Qwen3 document reranking (two-tool interface only)

Set `PI_SEARCH_RERANK_ENABLED=1`, `PI_SEARCH_FILTER_ENABLED=0`,
`PI_SEARCH_RERANK_BASE_URL=http://127.0.0.1:8152/v1`, and
`PI_SEARCH_RERANK_MODEL=qwen3-reranker-4b`. On each direct `search`, Piika retrieves
the top `REREANK_DEPTH` BM25 candidates (default 50; maximum 100), reads their
full documents, calls Qwen3-Reranker-4B `/score` with the **search query** and the
model's standard relevance instruction, sorts by descending activated score
(stable ties retain BM25 order), and returns the top `hits`. It does **not**
filter by a Yes/No threshold or invoke Qorrect. The visible result score is
the reranker probability; tool details retain per-candidate original BM25 scores.
Failed reads, invalid scores, and HTTP errors fail the search; no BM25 fallback.
This mode does not affect `pi-serini-3tool` or the separate Yes/No filter.
Filter and rerank cannot be enabled together.

### Optional Yes-score reranking with a generated question

To compare against an existing Qorrect Yes/No filter run while keeping its
question generator and predictor unchanged, keep `PI_SEARCH_FILTER_ENABLED=1`
and set `PI_SEARCH_FILTER_ACTION=rerank` (default: `filter`). Piika generates
the same per-search question and scores every BM25 candidate as before, but
**does not reject** scores below the threshold: it sorts all candidates by
descending `yesProbability` (stable BM25 ties), returning the top `hits`.
The search tool details include the generated question, score diagnostics, and
original BM25 scores under `rerank`. This is separate from reranking directly
on the raw query with `PI_SEARCH_RERANK_ENABLED=1`; never enable both modes at
once. `pi-serini-3tool` remains unchanged.

### Optional yes/no search filter (two-tool interface only)

Set `PI_SEARCH_FILTER_ENABLED=1` (or `true`) before launching the agent. Default is off;
`pi-serini-3tool` is unchanged even when this variable is set. Restart/reload the extension
when changing the flag. With `PI_SEARCH_FILTER_QUESTION_SOURCE=agent` (the default), enabled
`search` requires `filter_question`, a self-contained yes/no question tailored to the current
reasoning step, in addition to `reason`, `query`, and optional `hits`. Qorrect-generated and JEV
modes do not require an agent-provided `filter_question`.

```json
{"reason":"Find evidence for the candidate's education","query":"Ada Lovelace mathematics education","hits":10,"filter_question":"Does this document describe Ada Lovelace's mathematics education or tutors?"}
```

#### Qorrect question generation

Set `PI_SEARCH_FILTER_QUESTION_SOURCE=qorrect` to have a separate Qorrect **policy**
generate one filter question per search; `search` still requires `reason` and `query`,
but not `filter_question`. Piika's prompt uses `PI_SEARCH_ORIGINAL_QUESTION` (set by
the benchmark runner), the current `reason`, and the current `query`. It now matches
`reranq`'s frozen **full-output constrained** `prompts.py` snapshot exactly, including
rules that the question must be self-contained and must not rely on unresolved
references such as "my question". Previously seen `clue_docs` are **excluded by
default** to match training. To opt into out-of-training-distribution seen-document
context, explicitly set
`PI_SEARCH_FILTER_GENERATOR_INCLUDE_SEEN_DOCS=1` (only relevant with the clue-docs
interface). The default generator URL is `http://127.0.0.1:18086/v1` and its
model is `qorrect-decider-fulloutput-checkpoint717`; these can be overridden via
`PI_SEARCH_FILTER_GENERATOR_BASE_URL` and `PI_SEARCH_FILTER_GENERATOR_MODEL`.
The generation defaults are
64 output tokens, temperature 1, top-p 1 and top-k 0 with thinking disabled;
`PI_SEARCH_FILTER_GENERATOR_MAX_TOKENS` and `PI_SEARCH_FILTER_GENERATOR_TEMPERATURE`
can override the first two. The model cache alone does **not** start a server.
Piika passes the complete generated text (trimmed only at the ends) to the
scorer; it does not discard lines of a multi-line completion. For a service
that may reassign model aliases, set `PI_SEARCH_FILTER_GENERATOR_EXPECTED_MODEL_ROOT`
to the exact `/v1/models` `root`: Piika then checks the requested model's root
before **each** generated question and fails closed if it changes. Verify the
served alias's actual checkpoint before a new run; selecting the default alias alone
does not pin its root. The prompt alignment does **not** reproduce reranq's hard
decoding constraints or training-time multi-sample generation. Historical runs
retain their original prompt and checkpoint.

Qorrect only **writes questions**. Qwen (the default predictor) scores them:
`PI_SEARCH_FILTER_PREDICTOR=qwen`. JEV instead uses its own fixed question;
`PI_SEARCH_FILTER_PREDICTOR=jev` does not invoke Qorrect.
`PI_SEARCH_FILTER_BACKEND` and the Reranq filter are no longer supported.

#### pplx-decider-flash-v1.2 Yes-score rerank

Use the separately hosted, **non-vLLM** native decision service to score BM25
candidates. This option uses only the direct 2-tool `search` route and does not
change the 3-tool interface. With `PI_SEARCH_FILTER_ACTION=rerank`, Piika gets
BM25's top `REREANK_DEPTH` documents (default 50), sends each document in a
separate one-question decision request to `/score`, and returns the top `hits`
after sorting by the **raw true-minus-false logit margin**. Its calibrated
`P(true)` is logged separately; ties keep BM25 order. No probability threshold
removes candidates (`PI_SEARCH_FILTER_THRESHOLD` does not affect rerank).

For a fixed relevance instruction using each **current search query**:

```bash
PI_SEARCH_FILTER_ENABLED=1 PI_SEARCH_FILTER_ACTION=rerank \
PI_SEARCH_RERANK_ENABLED=0 PI_SEARCH_FILTER_PREDICTOR=pplx_decider \
PI_SEARCH_FILTER_QUESTION_SOURCE=general \
PI_SEARCH_FILTER_PREDICTOR_BASE_URL=http://127.0.0.1:18286 \
PI_SEARCH_FILTER_PREDICTOR_MODEL=pplx-decider-flash-v1.2 REREANK_DEPTH=50 \
npm run run:benchmark:query-set -- --benchmark bright-earthscience --query-set test --dry-run
```

For a **Qorrect-generated** Yes/No question per search, change
`PI_SEARCH_FILTER_QUESTION_SOURCE=qorrect` and set
`PI_SEARCH_FILTER_GENERATOR_BASE_URL` and `PI_SEARCH_FILTER_GENERATOR_MODEL`
to a running Qorrect policy. Qorrect writes the question; the decider scores
individual documents against it. Its affirmative-evidence instruction now matches
`reranq/decider_scorer.py` exactly (`pplx-decider-affirmative-evidence-v2` in
score details); the **general** search-query instruction remains separate. This
does not change document truncation: Piika still caps documents by characters,
unlike reranq's 512 whitespace-word training cap. Historical runs retain their original
instruction and results. `PI_SEARCH_FILTER_PREDICTOR_API` defaults to
`decider_score` for this predictor; incompatible API values fail at setup.
`PI_SEARCH_DECIDER_MAX_DOCUMENT_CHARS` defaults to 4000 (range 1–16000): this
is an explicit **character** prefix, not a 512-token cap; overruns of the
server's 8192-token context fail closed. HTTP or malformed-score failures
never fall back to BM25. `/score` is the custom decision service, **not** the
standard vLLM pooling API. See `runs/services/pplx-decider-flash-v1.2/README.md`
for the endpoint and model protocol. Agent final rankings still come from the
agent, not the reranked search output alone. This setting does not by itself
perform an offline replay of recorded trajectories.

`hits` is the maximum number returned after filtering; Piika scores BM25's top
`REREANK_DEPTH` candidates (default 50, capped at 100), keeps their original order,
and does not refill. A rejection can miss evidence later in a document, and zero
survivors do not prove absence. Failures do not silently bypass the filter.

Qwen defaults to `Qwen/Qwen3-Reranker-4B` with `/tokenize` + `/score` and lowercase
`yes`/`no` labels. Set `PI_SEARCH_FILTER_PREDICTOR_BASE_URL` (or
`PI_SEARCH_FILTER_BASE_URL`) to its running service; optionally override the served
model with `PI_SEARCH_FILTER_PREDICTOR_MODEL` (or `PI_SEARCH_FILTER_MODEL`).
Qwen also supports `PI_SEARCH_FILTER_API=generative_scoring` or
`vllm_prompt_logprobs`; JEV uses `systemone` and requires its own endpoint/key.
The endpoint may include `/v1`. Qwen keeps a document only when its activated Yes
probability exceeds the configured threshold (default 0.5); JEV keeps it only when
true probability exceeds false probability. Errors, missing documents, and invalid
responses fail closed. `/rerank` is not used.

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

This is still **filtering, not reranking**: Qwen keeps original retrieval order/scores,
uses the strict `score > threshold` rule and never refills. Different scoring models do
not necessarily have equally calibrated scores or retention rates.
512 tokens can cover different text under different tokenizers; for an offline replay
comparison, fix the same question, candidate set, and document text prefix. The replay
pipeline itself is not part of this backend switch.

Example DL19 exact-30 run with the deployed Qwen pooling service:

```bash
PI_SEARCH_FILTER_ENABLED=1 \
PI_SEARCH_FILTER_PREDICTOR=qwen \
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

Owns package-local prompt shaping and time-budget steering behavior. The benchmark runner passes
`PI_SEARCH_OUTPUT_MODE`, `PI_SEARCH_RANKED_LIST_DEPTH`, and `PI_SEARCH_RANKED_LIST_COUNT`
into each agent process so a submit-now steer at 70% of `TIMEOUT_SECONDS` uses the **same**
answer, ranked-list, or combined final-response sections as the original prompt. In
ranked-list mode it asks for known docids, never the answer-only Explanation/Exact Answer
format. The steer still blocks further retrieval; it cannot guarantee exact-30 when
there are fewer than 30 known valid docids.

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
