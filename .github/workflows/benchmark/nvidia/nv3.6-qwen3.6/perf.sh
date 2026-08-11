#!/bin/bash

# Copyright 2025-     FlagOS Contributors
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source ~/env.sh
source "${SCRIPT_DIR}/disable_local_proxy.sh"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[FATAL] CUDA_VISIBLE_DEVICES is unset." >&2
  exit 1
fi

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
for i in "${!GPU_IDS[@]}"; do
  GPU_IDS[$i]="${GPU_IDS[$i]//[[:space:]]/}"
  if [[ ! "${GPU_IDS[$i]}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] Invalid CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
done
GPU_DEVICE_LIST=$(IFS=,; echo "${GPU_IDS[*]}")

start=$(date +%s)

nvidia-smi -i "$GPU_DEVICE_LIST" -lgc 1830,1830
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,utilization.gpu --format=csv

numactl --cpunodebind=1 --membind=1 \
python3 ${SCRIPT_DIR}/all_perf.py --input-len=32768 --output-len=1024 --concurrency=16

numactl --cpunodebind=1 --membind=1 \
python3 ${SCRIPT_DIR}/all_perf.py --input-len=16384 --output-len=1024 --concurrency=16

numactl --cpunodebind=1 --membind=1 \
python3 ${SCRIPT_DIR}/all_perf.py --input-len=4096  --output-len=1024 --concurrency=16

nvidia-smi -i "$GPU_DEVICE_LIST" -rgc
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,utilization.gpu --format=csv

end=$(date +%s)
duration=$((end - start))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "[INFO] Benchmark elapsed time: ${minutes}m${seconds}s."
