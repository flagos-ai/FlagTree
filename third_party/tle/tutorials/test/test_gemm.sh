#!/usr/bin/env sh

# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
