# Reranker model services

Keep virtual environments, model weights, and logs outside the repository.
Launchers do not install dependencies. Model arguments accept Hugging Face repo IDs
or local checkpoint paths. With a repo ID, vLLM or the native SDK downloads the
weights at startup and reuses the Hugging Face cache on subsequent runs. Use a
local checkpoint path for offline deployment.

Run these commands from the repository root with the appropriate Python environment
activated. The GPU IDs below assume two GPUs; adjust them for your host.

## Qorrect (vLLM)

Use an external vLLM environment that supports the checkpoint's Qwen3.5 text
architecture. This is intended for a GPU host; Apple Silicon requires a suitable
backend such as vLLM Metal, whose compatibility is separate from this launcher.

```bash
bash scripts/rerankers/qorrect_server.sh \
  ricky42613/qorrect-rollout8-general-inject
```

If the model requires authentication, run `hf auth login` with an authorized
account before starting.

The first argument is the checkpoint path or Hugging Face ID. Use a local path
to avoid vLLM downloading weights. Defaults:

- API base: `http://127.0.0.1:8001/v1`
- Served model name: `qorrect`
- Context length: 2048 tokens (prompt plus output)
- Thinking disabled via `--default-chat-template-kwargs`
- `--generation-config vllm` avoids implicit checkpoint generation settings;
  the client explicitly sets output length and temperature.

Additional vLLM serve options follow the model argument and override launcher
defaults, for example `--port 9001 --max-model-len 4096`. `--help` forwards to
vLLM without loading a model. The launcher does not reproduce the checkpoint's
training-time special-token masks. It does not modify the existing local Mac
smoke-test deployment.

## Qwen3 reranker (vLLM pooling)

The intended checkpoint is **`Qwen/Qwen3-Reranker-4B`**, using the original
weights, not an already converted sequence-classification checkpoint. Requires vLLM support for the
Qwen3 reranker classification overrides and pooling score API.

```bash
PIIKA_QWEN_RERANKER_VLLM=/path/to/external/venv/bin/vllm \
  bash scripts/rerankers/qwen_reranker_server.sh /path/to/Qwen3-Reranker-4B
```

Defaults: loopback port `8003`, served model name `qwen3-reranker`, context length
8192, and `--runner pooling`. Additional vLLM options override defaults. The
launcher uses vLLM's `Qwen3ForSequenceClassification` overrides with
`classifier_from_token=["no","yes"]` and the bundled `qwen3_reranker.jinja`.
This follows the [official vLLM example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/qwen3_reranker_online.py);
the template is derived from its Apache-2.0 licensed
[template](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/template/qwen3_reranker.jinja).

The endpoint is `/score`, not chat completions:

```bash
curl http://127.0.0.1:8003/score \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-reranker","queries":"Who designed the analytical engine?","documents":["Charles Babbage designed the analytical engine."]}'
```

The TypeScript client is `src/pi-search/reranking/qwen_reranker.ts`. It sends
one `/score` request for all candidates, using the criterion question as the
query and an explicit instruction to judge whether each document satisfies it.
It requests activation and maps results by index, preserving candidate order.
The client is not yet wired into the search pipeline. GPU serving still needs
validation on the target host; launcher tests do not load model weights.

## Native Decider

Requires an external Python 3.12+ environment with the checkpoint's native
`pplx_decider` SDK and GPU dependencies installed. Defaults to CUDA and the
reference inference path (`fast=False`). PPLX Decider uses the native SDK,
**not vLLM**. Pass the Hugging Face ID directly; the SDK downloads the model.

```bash
python scripts/rerankers/decider_server.py \
  --model perplexity-ai/pplx-decider-flash-v1.2 \
  --host 127.0.0.1 --port 8002 \
  --max-batch-size 8 --max-wait-ms 10 --max-queue-size 256
```

Both services run in the foreground; start them in separate terminals. Check
readiness after model loading:

```bash
curl -f http://127.0.0.1:8001/v1/models  # Qorrect
curl -f http://127.0.0.1:8002/health     # PPLX Decider
```

- `GET /health`: readiness, available after model loading.
- `POST /v1/decisions`: text `state`, one named `noul` question, and a `model`
  matching `--served-model-name` (default `pplx-decider-flash-v1.2`).
- Responses use `model`, `answers`, and `usage`; `answers.<name>.noul` is the
  SDK's calibrated probability, not a raw logit margin.

```bash
curl http://127.0.0.1:8002/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pplx-decider-flash-v1.2","state":"Charles Babbage designed the analytical engine.","questions":{"relevance":{"type":"noul","instructions":"Does this document identify the designer of the analytical engine?"}}}'
```

Concurrent requests are queued for a single inference worker. A batch flushes
when full or when its oldest request reaches `max-wait-ms`; new arrivals do not
reset the deadline. This is a collection timeout, not an end-to-end latency
limit: earlier inference can delay the next batch. SDK token budgets may further
split a batch. A full queue returns HTTP 503. Client cancellation does not
interrupt already running GPU inference.

## Run BRIGHT Earth Science with Qorrect + PPLX

Run from the repository root after installing the `bright-earth-science`
benchmark manifest (`earth-science` query set and qrels) and registering
`brine/Qwen3.8-27B` in Pi. The Brine provider should reference the key with
`"apiKey": "!printenv BRINE_API_KEY"`, which works with the pinned Pi version.
Pyserini must serve `bright-earth-science` on port 8083; Qorrect and PPLX must
be ready on ports 8001 and 8002 respectively.

Start with one shard to validate service stability before increasing concurrency:

```bash
export BRINE_API_KEY=<your-key>
export PYSERINI_REST_BASE_URL=http://127.0.0.1:8083
export PYSERINI_REST_INDEX=bright-earth-science

npm run run:benchmark:query-set:sharded-shared-bm25 -- \
  --benchmark bright-earth-science \
  --query-set earth-science \
  --model brine/Qwen3.8-27B \
  --shard-count 1 \
  --timeout-seconds 900 \
  --output-mode ranked_list \
  --ranked-list-depth 30 \
  --ranked-list-count 30 \
  --tool-interface pyserini-rest-2tool \
  --reranking-config configs/reranking/qorrect-pplx.json \
  --output-root "runs/earth-science-qwen38-qorrect-pplx-$(date +%Y%m%d_%H%M%S)" \
  --no-auto-evaluate-on-merge
```

This runs the full query set, not a single-query smoke test. Append `--dry-run`
to inspect the plan without launching. Change `--shard-count 1` to `4` only after
verifying the services can sustain the extra load. The timeout is per query;
Qorrect and PPLX also have separate request timeouts. Exact 30 requests 30 unique
docids, but short outputs are flagged rather than padded. Check
`ranked_list_count_error` even when a query is marked `completed`.

See [reranking configuration and experiment options](../../configs/reranking/README.md)
for more details. Remote Pyserini manages its own threads; shard count controls
parallel query workers, not the retrieval server's thread count.

## Tests

Tests (no SDK or GPU required, Python 3.12+):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/rerankers -p 'test_*.py'
```
