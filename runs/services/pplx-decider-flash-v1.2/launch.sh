#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/home/ricky42613/piika
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
# Ubuntu's /usr/bin/python3.12 lacks development headers; same-version
# Conda Python headers are needed to build Triton's local CUDA driver shim.
export CPATH=/mnt/data/home/ricky42613/miniconda3/envs/vllm/include/python3.12${CPATH:+:$CPATH}
export TRITON_CACHE_DIR=/mnt/data/home/ricky42613/piika/runs/services/pplx-decider-flash-v1.2/triton-cache
export PPLX_DECIDER_PORT=18286
exec /mnt/data/home/ricky42613/.venvs/pplx-decider-flash-v1.2/bin/python -u runs/services/pplx-decider-flash-v1.2/server.py
