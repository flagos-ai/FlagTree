#!/bin/bash

rm -rf ~/.triton/cache

export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1

port=8333
# Check whether port is occupied
while ss -ltn | grep -q ":${port} "; do
    echo "Port ${port} is occupied, trying next..."
    port=$((port + 2))
done

echo "Using master_port=${port}"

run_test() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    torchrun \
        --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=${port} \
        "${script_dir}/test_tle_distributed_d2d.py"
}

run_test

if [ $? -ne 0 ]; then
    echo "ERROR: test_tle_distributed_d2d failed"
    exit 1
fi
