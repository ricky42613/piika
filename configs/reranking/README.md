# Per-search reranking configurations

## BRIGHT Earth Science: Qwen3.8 + Qorrect + PPLX

Run from the repository root. Prerequisites:

- The `bright-earth-science` benchmark manifest is installed, with the
  `earth-science` query set and qrels.
- Pi has `brine/Qwen3.8-27B` registered with the appropriate Brine API
  URL and `"apiKey": "!printenv BRINE_API_KEY"` (compatible with the pinned Pi version).
- Pyserini serves `bright-earth-science` on port 8083.
- Qorrect serves `qorrect` on port 8001, and native PPLX Decider serves
  `pplx-decider-flash-v1.2` on port 8002.

Full experiment command (Bash):

```bash
# Read the key without putting it in shell history.
read -rsp 'BRINE_API_KEY: ' BRINE_API_KEY; echo
export BRINE_API_KEY
export PYSERINI_REST_BASE_URL=http://127.0.0.1:8083
export PYSERINI_REST_INDEX=bright-earth-science

nohup npm run run:benchmark:query-set:sharded-shared-bm25 -- \
  --benchmark bright-earth-science \
  --query-set earth-science \
  --model brine/Qwen3.8-27B \
  --shard-count 4 \
  --timeout-seconds 900 \
  --output-mode ranked_list \
  --ranked-list-depth 30 \
  --ranked-list-count 30 \
  --tool-interface pyserini-rest-2tool \
  --reranking-config configs/reranking/qorrect-pplx.json \
  --output-root "runs/earth-science-qwen38-qorrect-pplx-$(date +%Y%m%d_%H%M%S)" \
  --no-auto-evaluate-on-merge &
```

Append `--dry-run` to inspect the launch plan without running the experiment.
Dry runs do not verify model authentication or service readiness.

- Four shards run query workers concurrently; timeout is 900 seconds per query.
- `--ranked-list-depth 30` sets the maximum; `--ranked-list-count 30` requests
  exactly 30 unique docids. Short outputs are flagged, not automatically padded.
- `--no-auto-evaluate-on-merge` disables automatic evaluation after merging.
- Remote Pyserini manages its own threads; `PI_BM25_THREADS` does not control it.

## Reranking configuration

`qorrect-pplx.json` uses customized Qorrect criteria (port 8001) and the native
PPLX Decider scorer (port 8002), with at least 50 candidates per search. Both URLs
are loopback addresses: start the services on the same host as Piika or change
the URLs. See [service deployment](../../scripts/rerankers/README.md).

Keep secrets out of these files. For authentication, use the supported `tokenEnv`
field to reference an environment variable rather than embedding a token.
Do not modify a configuration while a run or its retries are using it.
