# pplx-decider-flash-v1.2 local offline scorer

**Status:** running on `127.0.0.1:18286`, GPU 0. Not a generic vLLM
instance. The model's full-attention layers are noncausal and its separate
255-way readout is applied at each inline marker; standard vLLM `/score`
would score a different model. The server uses the model repository's
`pplx_decider.Decider` reference implementation (`fast=False`).

- Isolated environment: `/mnt/data/home/ricky42613/.venvs/pplx-decider-flash-v1.2`
- Upstream dependency pins: `requirements.upstream.txt` (torch 2.14.0,
  transformers 5.17.0)
- HF revision: `9ff7cf9838e59df461c276b7699d56b244d2af58`
- Model: `model` symlink to revision-pinned HF cache; no other service altered.
- Python 3.12 from Ubuntu lacks Python.h; `launch.sh` explicitly sets CPATH
  to the same-version Conda Python 3.12 headers for Triton JIT.
- Reference path, maximum 8192 tokens per request, batch up to 32 independent
  documents, 4096 padded batch-token budget. Inputs exceeding the model limit
  raise an error rather than silently truncating. For long documents, cap the
  text in the *offline scorer client* and document that cap.

Endpoints:

```bash
curl http://127.0.0.1:18286/health
curl http://127.0.0.1:18286/v1/models
curl -sS http://127.0.0.1:18286/score \
  -H 'Content-Type: application/json' \
  -d '{"model":"pplx-decider-flash-v1.2", "queries":"What causes a solar halo?", "documents":["Ice crystals refract sunlight."], "instruction":"Does this document provide evidence answering the search query?"}'
```

`/score` returns `data[].score` = calibrated P(true), **`data[].margin` =
true minus false raw decision logits** (prefer the margin for reranking to
avoid saturated-probability ties). Every candidate is sent as a separate
one-question decision request in the same batch, never as 50 questions in a
joint sequence. `/v1/decide` accepts `{"state":...,"questions":...}` for the
checkpoint's native protocol and returns `{"model":...,"response":...}`.
These are custom HTTP endpoints, not vLLM's OpenAI pooling `/score` semantics.

Smoke tests `score_smoke.json` and `decide_smoke.json` exercised real GPU
inference. Relevant/irrelevant examples scored 0.9683/0.0040 respectively.
Model currently uses safe reference fallback kernels; this is correct but may
be slower than the optional fast path. Do not claim measured throughput or
benchmark performance from the smoke test.

Start after stopping:

```bash
nohup bash runs/services/pplx-decider-flash-v1.2/launch.sh \
  > runs/services/pplx-decider-flash-v1.2/server.log 2>&1 < /dev/null &
echo $! > runs/services/pplx-decider-flash-v1.2/server.pid
```

Stop only this process after verifying ownership and PID:

```bash
kill -TERM "$(<runs/services/pplx-decider-flash-v1.2/server.pid)"
```
