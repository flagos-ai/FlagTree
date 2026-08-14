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

echo "[INFO] vLLM $(python3 -m pip show vllm |grep Version)"
echo "[INFO] Torch $(python3 -m pip show torch |grep Version)"
echo "[INFO] FlagGems $(python3 -m pip show flag_gems |grep Version)"
echo "[INFO] FlagTree $(python3 -m pip show flagtree |grep Version)"

PID_FILE="pid.txt"
if [[ -f "$PID_FILE" ]]; then
    pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARNING] Service already running: pid = $pid."
            bash ${SCRIPT_DIR}/stop.sh
        fi
    fi
fi
bash "${SCRIPT_DIR}/clear_fuser_process.sh"

start=$(date +%s)

export VLLM_USE_MODELSCOPE=true
export USE_FLAGGEMS=1
export USE_RESHAPE_AND_CACHE_FLASH=1

numactl --cpunodebind=1 --membind=1 \
nohup vllm serve ./Qwen3.6-27B/  \
    --tensor-parallel-size 2 \
    --port 8000  \
    --served-model-name qwen36 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --block-size 256  \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 2>&1 >vllm.log &
echo "$!" >pid.txt

PID_FILE="pid.txt"
if [[ ! -f "$PID_FILE" ]]; then
    echo "[FATAL] $PID_FILE not found!"
    exit 1
fi

pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] PID = $pid is invalid!"
    exit 1
fi

max_retry=180
for ((i=1; i<=max_retry; i++)); do
    sleep 10
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[FATAL] Process $pid does not exist, service startup failed!"
        exit 1
    fi
    if bash ${SCRIPT_DIR}/ping.sh 2>/dev/null; then
        echo ""
        echo "[INFO] Service startup successfully."
        break
    fi
    end=$(date +%s)
    duration=$((end - start))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    echo "[INFO] Service startup elapsed time: ${minutes}m${seconds}s"
    if (( i > max_retry )); then
        echo "[FATAL] Detection failed! maximum retry count reached: ${max_retry}"
        exit 1
    fi
done
