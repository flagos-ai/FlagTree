#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on one node with at least two GPUs:
#
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NPROC_PER_NODE=8 \
# MASTER_PORT=29581 \
# COMPARE_FLAGGEMS=/path/to/FlagGems/src/flag_gems/fused/fused_rs_norm_ag.py \
# bash 10-intra-node-rs-norm-ag.sh 2>&1
#
# MASTER_ADDR defaults to localhost; MASTER_PORT defaults to 29501.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec torchrun \
    --nproc-per-node="${NPROC_PER_NODE:-4}" \
    --nnodes=1 \
    --node-rank=0 \
    --master-addr="${MASTER_ADDR:-localhost}" \
    --master-port="${MASTER_PORT:-29501}" \
    "$script_dir/10-intra-node-rs-norm-ag.py" "$@"
