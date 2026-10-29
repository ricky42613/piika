#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM="${PIIKA_QWEN_RERANKER_VLLM:-vllm}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 ORIGINAL_QWEN3_RERANKER_PATH_OR_ID [vLLM serve options...]" >&2
  exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  exec "$VLLM" serve --help
fi
if [[ "$1" == -* ]]; then
  echo "The first argument must be a model path or Hugging Face ID." >&2
  exit 2
fi
MODEL="$1"
shift

# Original Qwen3 reranker weights, not an already converted seq-cls checkpoint.
# See vLLM examples/pooling/score/qwen3_reranker_online.py.
exec "$VLLM" serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8003 \
  --served-model-name qwen3-reranker \
  --max-model-len 8192 \
  --runner pooling \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template "$SCRIPT_DIR/qwen3_reranker.jinja" \
  "$@"
