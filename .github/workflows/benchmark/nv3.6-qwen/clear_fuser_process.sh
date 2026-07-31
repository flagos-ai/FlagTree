#!/bin/bash
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
  DEVICES+=("/dev/nvidia${gpu_id}")
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
