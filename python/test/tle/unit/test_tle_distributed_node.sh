#!/usr/bin/env bash
set -euo pipefail

# This is a multi-node test and must be started once on every node.
# Run from the repository root, for example:
#   # node 0
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash python/test/tle/unit/test_tle_distributed_node.sh
#   # node 1
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#       bash python/test/tle/unit/test_tle_distributed_node.sh
#
# MASTER_ADDR must be the reachable address of node 0. Both nodes must use
# the same NNODES, NPROC_PER_NODE, MASTER_ADDR and MASTER_PORT.
# The test requires NNODES >= 2. NNODES defaults to 2; NPROC_PER_NODE
# defaults to gpu (all visible GPUs).
# MASTER_PORT defaults to 29501.

export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_bond_0
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nproc_per_node="${NPROC_PER_NODE:-gpu}"
nnodes="${NNODES:-2}"
node_rank="${NODE_RANK:-0}"
master_port="${MASTER_PORT:-29501}"

if [[ "${nnodes}" -gt 1 ]]; then
    master_addr="${MASTER_ADDR:?The multi-node runtime must set MASTER_ADDR (the reachable IP or hostname of node 0)}"
else
    master_addr="${MASTER_ADDR:-localhost}"
fi

if [[ "${node_rank}" -lt 0 || "${node_rank}" -ge "${nnodes}" ]]; then
    echo "NODE_RANK=${node_rank} is out of bounds. Valid range is 0 to $((nnodes - 1))" >&2
    exit 1
fi

printf "Starting TLE distributed node test: node_rank=%s, nproc_per_node=%s, master=%s:%s\n" "${node_rank}" "${nproc_per_node}" "${master_addr}" "${master_port}"

exec torchrun --nproc-per-node="${nproc_per_node}" --nnodes="${nnodes}" --node-rank="${node_rank}" --master-addr="${master_addr}" --master-port="${master_port}" "${script_dir}/test_tle_distributed_node.py"
