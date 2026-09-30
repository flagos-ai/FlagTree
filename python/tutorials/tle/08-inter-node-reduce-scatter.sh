#!/usr/bin/env bash
set -euo pipefail

# Run once on each node from the repository root:
#
#     # Node 0
#     NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
#         bash python/tutorials/tle/07-inter-node-reduce-scatter.sh
#     # Node 1
#     NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
#         bash python/tutorials/tle/07-inter-node-reduce-scatter.sh
#
# Use matching topology, dimensions, dtype, and tuning settings on every node.
# MASTER_ADDR must be reachable from all nodes; MASTER_PORT defaults to 29501.

# Set CUDA_VISIBLE_DEVICES externally to select GPUs.
export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_bond_0
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nproc_per_node="${NPROC_PER_NODE:-gpu}"
nnodes="${NNODES:-2}"
master_port="${MASTER_PORT:-29501}"

if [[ ! "$nnodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "NNODES must be a positive integer" >&2
    exit 1
fi
if [[ "$nnodes" -gt 1 ]]; then
    node_rank="${NODE_RANK:?Set NODE_RANK on every node}"
    master_addr="${MASTER_ADDR:?Set MASTER_ADDR to the reachable address of node 0}"
else
    node_rank="${NODE_RANK:-0}"
    master_addr="${MASTER_ADDR:-localhost}"
fi
if [[ ! "$node_rank" =~ ^(0|[1-9][0-9]*)$ ]] || [[ "$node_rank" -ge "$nnodes" ]]; then
    echo "NODE_RANK must be in [0, NNODES)" >&2
    exit 1
fi

exec torchrun \
    --nproc-per-node="$nproc_per_node" \
    --nnodes="$nnodes" \
    --node-rank="$node_rank" \
    --master-addr="$master_addr" \
    --master-port="$master_port" \
    "$script_dir/07-inter-node-reduce-scatter.py" "$@"
