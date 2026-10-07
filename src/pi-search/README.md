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
when changing the flag. With `PI_SEARCH_FILTER_QUESTION_SOURCE=agent` (the default), enabled
`search` requires `filter_question`, a self-contained yes/no question tailored to the current
reasoning step, in addition to `reason`, `query`, and optional `hits`. Qorrect-generated and JEV
modes do not require an agent-provided `filter_question`.

```json
{"reason":"Find evidence for the candidate's education","query":"Ada Lovelace mathematics education","hits":10,"filter_question":"Does this document describe Ada Lovelace's mathematics education or tutors?"}
```

#### Qorrect rollout-4 question generation

Set `PI_SEARCH_FILTER_QUESTION_SOURCE=qorrect` to have the separate
`ricky42613/qorrect-rollout4` **policy** generate one filter question per search;
`search` still requires `reason` and `query`, but not `filter_question`. The rollout-4
prompt uses `PI_SEARCH_ORIGINAL_QUESTION` (set by the benchmark runner), the current
`reason`, and the current `query`. It matches the model repository's `prompts.py`;
previously seen `clue_docs` are **excluded by default** to match training. To opt into
out-of-training-distribution seen-document context, explicitly set
`PI_SEARCH_FILTER_GENERATOR_INCLUDE_SEEN_DOCS=1` (only relevant with the clue-docs
interface). Use `PI_SEARCH_FILTER_GENERATOR_BASE_URL` for the running vLLM
`/v1/completions` server, and override `PI_SEARCH_FILTER_GENERATOR_MODEL` if the served
model alias differs from `ricky42613/qorrect-rollout4`. The generation defaults are
64 output tokens, temperature 1, top-p 1 and top-k 0 with thinking disabled;
`PI_SEARCH_FILTER_GENERATOR_MAX_TOKENS` and `PI_SEARCH_FILTER_GENERATOR_TEMPERATURE`
can override the first two. The model cache alone does **not** start a server.

Qorrect only **writes questions**. Qwen (the default predictor) scores them:
`PI_SEARCH_FILTER_PREDICTOR=qwen`. JEV instead uses its own fixed question;
`PI_SEARCH_FILTER_PREDICTOR=jev` does not invoke Qorrect.
`PI_SEARCH_FILTER_BACKEND` and the Reranq filter are no longer supported.

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
