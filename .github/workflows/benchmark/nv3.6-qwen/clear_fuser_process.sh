#!/bin/bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES 未设置，跳过 GPU 进程清理"
  exit 0
fi

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
DEVICES=()
for gpu_id in "${GPU_IDS[@]}"; do
  # 允许逗号两侧包含空白，但设备编号必须是非负整数。
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "无效的 CUDA_VISIBLE_DEVICES：$CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
  DEVICES+=("/dev/nvidia${gpu_id}")
done

# 获取占用设备的 PID
mapfile -t PIDS < <(
  fuser "${DEVICES[@]}" 2>/dev/null \
  | grep -oE '[0-9]+' \
  | sort -u
)

if ((${#PIDS[@]} == 0)); then
  echo "没有发现占用 ${DEVICES[*]} 的进程"
  exit 0
fi

echo "将终止以下进程：${PIDS[*]}"
ps -fp "${PIDS[@]}" || true
kill "${PIDS[@]}" 2>/dev/null || true
sleep 5

# 仍存在则强制终止
mapfile -t REMAINING < <(
  fuser "${DEVICES[@]}" 2>/dev/null \
  | grep -oE '[0-9]+' \
  | sort -u
)

if ((${#REMAINING[@]} > 0)); then
  echo "仍占用设备，强制终止：${REMAINING[*]}"
  kill -9 "${REMAINING[@]}" 2>/dev/null || true
fi

echo "当前占用情况："
fuser -v "${DEVICES[@]}" || true