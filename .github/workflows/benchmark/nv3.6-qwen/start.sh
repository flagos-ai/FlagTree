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

source ~/env.sh
source ${BENCH_SCRIPT_DIR}/disable_local_proxy.sh

python3 -m pip show flag_gems |grep Version
python3 -m pip show flagtree |grep Version

PID_FILE="pid.txt"
if [[ -f "$PID_FILE" ]]; then
    pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARNING] Service already running: pid = $pid."
            bash ${BENCH_SCRIPT_DIR}/stop.sh
        fi
    fi
fi
bash ${BENCH_SCRIPT_DIR}/clear_fuser_process.sh

start=$(date +%s)

export VLLM_USE_MODELSCOPE=true
export USE_FLAGGEMS=1
export USE_RESHAPE_AND_CACHE_FLASH=1

numactl --cpunodebind=1 --membind=1 \
nohup vllm serve ./Qwen3.6-35B-A3B-nomtp/  \
    --tensor-parallel-size 4 \
    --port 8000  \
    --served-model-name qwen36 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --block-size 256  \
    --gpu-memory-utilization 0.7 \
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
    if bash ping.sh 2>/dev/null; then
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
