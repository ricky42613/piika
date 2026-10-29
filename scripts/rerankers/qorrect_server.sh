#!/usr/bin/env bash
set -euo pipefail

# Point to vllm in an external environment; this launcher installs nothing.
VLLM="${PIIKA_QORRECT_VLLM:-vllm}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 MODEL_PATH_OR_ID [vLLM serve options...]" >&2
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

# Preserve cwd for relative checkpoint paths. Trailing options can override defaults.
exec "$VLLM" serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qorrect \
  --max-model-len 2048 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  "$@"
