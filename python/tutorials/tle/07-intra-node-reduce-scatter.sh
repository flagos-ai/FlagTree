#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on one node with eight GPUs:
#
#     M=2048 N=2048 DTYPE=bf16 \
#       bash python/tutorials/tle/07-intra-node-reduce-scatter.sh
#
# Override CUDA_VISIBLE_DEVICES or the FlagCX NIC list when needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FLAGCX_IB_HCA="${FLAGCX_IB_HCA:-mlx5_0,mlx5_1,mlx5_bond_0}"
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ "${CLEAR_TRITON_CACHE:-0}" == "1" ]]; then
    rm -rf "${TRITON_CACHE_DIR:-${HOME}/.triton/cache}"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-8333}"

exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/07-intra-node-reduce-scatter.py" "$@"
