#!/usr/bin/env bash
# Before running this script, it is necessary to set the environment variables 
# NNODES, NODE_RANK, MASTER_ADDR and MASTER_PORT for the distributed nodes.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FLAGCX_IB_HCA="${FLAGCX_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_ADDR="${MASTER_ADDR:-10.0.9.3}"


nproc_per_node="${NPROC_PER_NODE:-2}"
nnodes="${NNODES:-2}"
node_rank="${NODE_RANK:-0}"
master_port="${MASTER_PORT:-8352}"

if [[ "${nnodes}" -gt 1 ]]; then
    master_addr="${MASTER_ADDR:?The multi-node runtime must set MASTER_ADDR (the reachable IP or hostname of node 0)}"
else
    master_addr="${MASTER_ADDR:-localhost}"
fi

if [[ "${node_rank}" -lt 0 || "${node_rank}" -ge "${nnodes}" ]]; then
    echo "NODE_RANK=${node_rank} is out of bounds. Valid range is 0 to $((nnodes - 1))" >&2
    exit 1
fi

echo "Starting TLE signal test: node_rank=${node_rank}/${nnodes}, "
echo "nproc_per_node=${nproc_per_node}, master=${master_addr}:${master_port}"

torchrun \
    --nproc_per_node="${nproc_per_node}" \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    "${script_dir}/test_tle_signal.py"