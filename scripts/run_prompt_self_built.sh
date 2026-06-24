#!/usr/bin/env bash
set -euo pipefail

SEARCH_SCRIPT="${SEARCH_SCRIPT:-/mnt/data/home/ricky42613/polar-piika/test-time-tool-optimization/hybrid_search/search.py}"

npm run run:prompt -- \
  --query "Who wrote the novel The Count of Monte Cristo?" \
  --search-script "$SEARCH_SCRIPT" \
  --timeout-seconds 900
