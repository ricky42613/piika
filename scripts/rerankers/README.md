# Reranker model services

Keep virtual environments, model weights, and logs outside the repository.
Launchers do not install dependencies or download models themselves. Pass a local
checkpoint path to avoid the native SDK downloading a Hugging Face repo at startup.

## Qorrect (vLLM)

Use an external vLLM environment that supports the checkpoint's Qwen3.5 text
architecture. This is intended for a GPU host; Apple Silicon requires a suitable
backend such as vLLM Metal, whose compatibility is separate from this launcher.

```bash
PIIKA_QORRECT_VLLM=/path/to/external/venv/bin/vllm \
  bash scripts/rerankers/qorrect_server.sh /path/to/qorrect-checkpoint
```

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
reference inference path (`fast=False`).

```bash
PIIKA_DECIDER_PYTHON=/path/to/external/venv/bin/python \
  bash scripts/rerankers/decider_server.sh \
  --model /path/to/pplx-decider-flash-v1.2 \
  --host 127.0.0.1 --port 8002 \
  --max-batch-size 8 --max-wait-ms 10 --max-queue-size 256
```

`PIIKA_DECIDER_PYTHON` defaults to `python3`. Relative checkpoint paths are
resolved from the caller's working directory. Run `--help` to inspect options
without loading the SDK or model. The launcher runs in the foreground; Ctrl-C
stops it. No authentication is provided: use loopback or a trusted proxy, not a
publicly exposed listener.

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

Tests (no SDK or GPU required, Python 3.12+):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/rerankers -p 'test_*.py'
```
