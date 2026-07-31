#!/bin/bash

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
    echo "进程 $pid 不存在或无权限访问"
    exit 1
else
    echo "正在终止进程：$pid"

    if kill "$pid"; then
        # 等待进程退出，最多等待 60 秒
        for ((i=1; i<=6; i++)); do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "进程 $pid 已成功退出"
                break
            fi
            sleep 10
        done

        # 进程仍存在，强制终止
        if kill -0 "$pid" 2>/dev/null; then
            echo "进程 $pid 未退出，执行强制 kill"
            kill -9 "$pid"
            sleep 5
            if kill -0 "$pid" 2>/dev/null; then
                echo "失败：无法终止进程 $pid"
            else
                echo "进程 $pid 已被强制终止"
            fi
        fi
    else
        echo "失败：kill 进程 $pid 失败"
    fi
fi

echo
if pgrep -af 'vllm' > /dev/null; then
    echo "[WARNING] 仍存在 vLLM 进程："
    pgrep -af 'vllm'
else
    echo "[INFO] 未发现 vLLM 进程，服务停止成功，符合预期"
fi
