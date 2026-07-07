#!/bin/bash

rm -rf ~/.triton/cache

export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

run_test() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    torchrun \
        --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=8333 \
        "${script_dir}/test_tle_distributed_d2d.py"
}

run_test

if [ $? -ne 0 ]; then
    echo "ERROR: test_tle_distributed_d2d failed"
    exit 1
fi
