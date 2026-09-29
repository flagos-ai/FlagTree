#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on one node with at least two GPUs:
#
#     NPROC_PER_NODE=4 M=2048 N=2048 DTYPE=bf16 \
#         bash python/tutorials/tle/06-intra-node-reduce-scatter.sh
#
# MASTER_ADDR defaults to localhost; MASTER_PORT defaults to 29501.

export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nproc_per_node="${NPROC_PER_NODE:-4}"
nnodes="${NNODES:-1}"
master_port="${MASTER_PORT:-29501}"

if [[ ! "$nnodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "NNODES must be a positive integer" >&2
    exit 1
fi
if [[ "$nnodes" -ne 1 ]]; then
    echo "This tutorial requires NNODES=1" >&2
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
    "$script_dir/06-intra-node-reduce-scatter.py" "$@"
