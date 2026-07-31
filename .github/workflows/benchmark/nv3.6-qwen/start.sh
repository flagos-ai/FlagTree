#!/bin/bash

source ~/env.sh
source ${BENCH_SCRIPT_DIR}/disable_proxy.sh

python3 -m pip show flag_gems |grep Version
python3 -m pip show flagtree |grep Version

PID_FILE="pid.txt"
if [[ -f "$PID_FILE" ]]; then
    pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        if kill -0 "$pid" 2>/dev/null; then
            echo "已有服务启动，进程 $pid"
            bash stop.sh
        fi
    fi
fi
bash ${BENCH_SCRIPT_DIR}/clear_fuser_process.sh

start=$(date +%s)

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
    echo "错误：找不到 $PID_FILE"
    exit 1
fi

pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "错误：PID 无效：$pid"
    exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
    echo "进程 $pid 不存在或无权限访问，服务启动失败"
    exit 1
fi

max_retry=180
for ((i=1; i<=max_retry; i++)); do
    sleep 10
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "进程 $pid 不存在或无权限访问，服务启动失败"
        exit 1
    fi
    if bash ping.sh 2>/dev/null; then
        echo ""
        echo "检测到服务启动成功"
        break
    fi
    end=$(date +%s)
    duration=$((end - start))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    echo "服务启动已耗时: ${minutes}分${seconds}秒"
    if (( i > max_retry )); then
        echo "检测失败，已达到最大重试次数：${max_retry}"
        exit 1
    fi
done
