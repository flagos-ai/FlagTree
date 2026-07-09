#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PARENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

echo "$SCRIPT_DIR"

python "${PARENT_DIR}/tle_hopper_fa_ws_pipelined_pingpong_persistent.py" \
  --warmup 25 \
  --rep 100 \
  --block-m 128 \
  --block-n 128 \
  --cuda-graph \
  --out "${SCRIPT_DIR}/tle_fa_user_promise_benchmark.csv" \
  --problem 4x32x1024x128 \
  --problem 4x32x2048x128 \
  --problem 4x32x4096x128 \
  --problem 4x32x8192x128 \
  --check \
  --include-sdpa \
  --sdpa-requires-grad \
  --sm-scale 1.3 \
  --dump-summary \
  "$@"
