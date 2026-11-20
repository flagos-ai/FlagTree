#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PARENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

echo "$SCRIPT_DIR"

python "${PARENT_DIR}/tle_hopper_gemm_ws_persistent.py" \
  --compare \
  --warmup 25 \
  --rep 100 \
  --bm 128 \
  --bn 128 \
  --bk 64 \
  --cuda-graph \
  --out "${SCRIPT_DIR}/tle_gemm_user_promise_benchmark.csv" \
  --shape 2048x2048x2048 \
  --shape 4096x4096x4096 \
  --shape 8192x8192x512 \
  --shape 8192x8192x8192 \
  --check \
  "$@"
