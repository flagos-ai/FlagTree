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

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[WARNING] CUDA_VISIBLE_DEVICES is unset; skipping GPU cleanup."
  exit 0
fi

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
DEVICES=()
for gpu_id in "${GPU_IDS[@]}"; do
  # Allow whitespace; require non-negative integer IDs.
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] Invalid CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
  DEVICES+=("/dev/alixpu_ppu${gpu_id}")
done

# Find PIDs using the devices.
mapfile -t PIDS < <(
  fuser "${DEVICES[@]}" 2>/dev/null \
  | grep -oE '[0-9]+' \
  | sort -u
)

if ((${#PIDS[@]} == 0)); then
  echo "[INFO] No processes found on ${DEVICES[*]}."
  exit 0
fi

echo "[INFO] Terminating processes: ${PIDS[*]}."
ps -fp "${PIDS[@]}" || true
kill "${PIDS[@]}" 2>/dev/null || true
sleep 5

# Force-kill remaining processes.
mapfile -t REMAINING < <(
  fuser "${DEVICES[@]}" 2>/dev/null \
  | grep -oE '[0-9]+' \
  | sort -u
)

if ((${#REMAINING[@]} > 0)); then
  echo "[WARNING] Force-killing processes: ${REMAINING[*]}"
  kill -9 "${REMAINING[@]}" 2>/dev/null || true
fi

echo "[INFO] Current usage:"
fuser -v "${DEVICES[@]}" || true
