#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use an external environment with Python 3.12+ and the native pplx_decider SDK.
PYTHON="${PIIKA_DECIDER_PYTHON:-python3}"

# Preserve the caller's working directory so relative checkpoint paths work.
# exec forwards signals and the exit status without an extra shell process.
exec "$PYTHON" -u -B "$SCRIPT_DIR/decider_server.py" "$@"
