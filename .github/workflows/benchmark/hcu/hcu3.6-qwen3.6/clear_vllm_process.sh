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
