#!/bin/bash

if [ "$1" = "debug" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=all
else
    unset NCCL_DEBUG
    unset NCCL_DEBUG_SUBSYS
fi

export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

nproc_per_node=${NPROC_PER_NODE:-2}
nnodes=${NNODES:-2}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-10.0.9.3}
port=${MASTER_PORT:-8335}

if [ "${nnodes}" -eq 1 ]; then
    while ss -ltn | grep -q ":${port} "; do
        echo "Port ${port} is occupied, trying next..."
        port=$((port + 2))
    done
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Using master ${master_addr}:${port}, node ${node_rank}/${nnodes}"

torchrun \
    --nproc_per_node="${nproc_per_node}" \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${port}" \
    "${script_dir}/test_tle_get_node_rank.py"
