#!/bin/bash

PID_FILE="pid.txt"
if [[ ! -f "$PID_FILE" ]]; then
    echo "[ERROR] $PID_FILE not found!"
    exit 1
fi

pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] PID = $pid is invalid!"
    exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
    echo "[ERROR] Process $pid does not exist, no need to stop."
    exit 1
else
    echo "[INFO] Stopping process: $pid"

    if kill "$pid"; then
        # Wait for process to exit, up to 60 seconds.
        for ((i=1; i<=6; i++)); do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[INFO] Process $pid has been successfully terminated."
                break
            fi
            sleep 10
        done

        # Process still exists, force kill.
        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARNING] Process $pid did not exit, force kill!"
            kill -9 "$pid"
            sleep 5
            if kill -0 "$pid" 2>/dev/null; then
                echo "[FATAL] Failed to terminate process $pid!"
            else
                echo "[INFO] Process $pid has been force terminated."
            fi
        fi
    else
        echo "[ERROR] Failed to kill process $pid!"
    fi
fi

echo
if pgrep -af 'vllm' > /dev/null; then
    echo "[WARNING] vLLM process still exists:"
    pgrep -af 'vllm'
else
    echo "[INFO] vLLM process not found, service stopped successfully, as expected."
fi
