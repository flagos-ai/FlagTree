#!/bin/bash
set -euo pipefail

DEVICE="/dev/nvidia4"

# 获取占用设备的 PID
mapfile -t PIDS < <(
  fuser "$DEVICE" 2>/dev/null \
  | grep -oE '[0-9]+' \
  | sort -u
)

if ((${#PIDS[@]} == 0)); then
  echo "没有发现占用 $DEVICE 的进程"
  exit 0
fi

echo "将终止以下进程：${PIDS[*]}"
ps -fp "${PIDS[@]}" || true
kill "${PIDS[@]}" 2>/dev/null || true
sleep 5

# 仍存在则强制终止
mapfile -t REMAINING < <(
  fuser "$DEVICE" 2>/dev/null \
  | grep -oE '[0-9]+' \
  | sort -u
)

if ((${#REMAINING[@]} > 0)); then
  echo "仍占用设备，强制终止：${REMAINING[*]}"
  kill -9 "${REMAINING[@]}" 2>/dev/null || true
fi

echo "当前占用情况："
fuser -v "$DEVICE" || true