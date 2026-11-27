#!/usr/bin/env bash
set -euo pipefail

# Run on both nodes with NODE_RANK=0/1 and the same reachable MASTER_ADDR.
# --nnodes=2 and --nproc-per-node=8 start sixteen workers across the two nodes.
# Adjust the FlagCX NIC list and visible devices as needed for the target machines.
export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_bond_0
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-1}"
# Each node's eight workers require at least eight visible GPUs and use the
# first eight entries in this visibility list, one GPU per worker.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

node_rank="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
master_addr="${MASTER_ADDR:?Set MASTER_ADDR to the reachable address of node 0}"
master_port="${MASTER_PORT:-29500}"
if [[ "$node_rank" != 0 && "$node_rank" != 1 ]]; then
    echo "NODE_RANK must be 0 or 1" >&2
    exit 1
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec torchrun \
    --nnodes=2 \
    --node-rank="$node_rank" \
    --nproc-per-node=8 \
    --master-addr="$master_addr" \
    --master-port="$master_port" \
    "$script_dir/06-inter-node-allgather.py"
