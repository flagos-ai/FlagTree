#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_MEM_ENABLE=1
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3


if [[ "${CLEAR_TRITON_CACHE:-0}" == "1" ]]; then
    rm -rf "${TRITON_CACHE_DIR:-${HOME}/.triton/cache}"
fi

NPROC_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=8333


torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/05-intra-node-allgather.py"
