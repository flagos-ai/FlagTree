#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on one node with at least two GPUs:
#
#     NPROC_PER_NODE=2 M=1024 N=4096 K=2048 \
#         bash python/tutorials/tle/10-fused-scaled-matmul-reduce-scatter.sh
#
# Use --scatter-dim 1 to reduce-scatter columns and --bias to add a bias.
# MASTER_ADDR defaults to localhost; MASTER_PORT defaults to 29501.
# CUDA_VISIBLE_DEVICES must include at least NPROC_PER_NODE GPUs.

# Point this at your FlagCX install if it lives elsewhere.
export FLAGCX_MODULE_PATH="${FLAGCX_MODULE_PATH:-/root/.flagtree/flagcx}"
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nproc_per_node="${NPROC_PER_NODE:-2}"
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

node_rank="${NODE_RANK:-0}"
master_addr="${MASTER_ADDR:-localhost}"

exec torchrun \
    --nproc-per-node="$nproc_per_node" \
    --nnodes="$nnodes" \
    --node-rank="$node_rank" \
    --master-addr="$master_addr" \
    --master-port="$master_port" \
    "$script_dir/10-fused-scaled-matmul-reduce-scatter.py" "$@"
