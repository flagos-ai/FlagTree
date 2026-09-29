#!/usr/bin/env bash

set -euo pipefail

export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

NODE_RANK=0
NNODES=2
NPROC_PER_NODE=4
MASTER_ADDR=10.0.9.3
MASTER_PORT=29500
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/06-inter-node-allgather.py"
