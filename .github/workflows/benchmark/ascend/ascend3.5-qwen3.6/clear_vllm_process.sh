#!/bin/bash

TARGET_DIR="$(readlink -f "${HOME}/flagrelease/qwen3.6")"

ps afx | grep "VLLM::EngineCore" | grep -v grep | awk '{print $1}' | while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue

    process_dir=$(pwdx "$pid" 2>/dev/null) || continue
    process_dir=${process_dir#*: }

    case "$process_dir" in
        "$TARGET_DIR"|"$TARGET_DIR"/*)
            echo "[INFO] Cleanup VLLM::EngineCore process $pid in $process_dir"
            kill "$pid"

            for ((attempt=1; attempt<=12; attempt++)); do
                sleep 5
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "[INFO] Process $pid stopped successfully"
                    break
                fi
                echo "[INFO] Waiting for process $pid to stop (${attempt}/12)"
            done

            if kill -0 "$pid" 2>/dev/null; then
                echo "[WARNING] Process $pid did not stop within 60 seconds, forcing termination"
                kill -9 "$pid"
            fi
            ;;
    esac
done