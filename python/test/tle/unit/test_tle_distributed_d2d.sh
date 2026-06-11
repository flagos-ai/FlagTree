rm -rf ~/.triton/cache
#!/bin/bash

# Check if the debug flag is provided as an argument
if [ "$1" == "debug" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=all
    echo "NCCL debug information enabled."
else
    unset NCCL_DEBUG
    unset NCCL_DEBUG_SUBSYS
    echo "NCCL debug information disabled."
fi
#export FLAGCX_SOCKET_IFNAME=eth0
export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Need to preload customized gloo library specified for FlagCX linkage
#export LD_PRELOAD=/usr/local/lib/libgloo.so
#export LD_PRELOAD=/usr/local/nccl/build/lib/libnccl.so
#export TORCH_DISTRIBUTED_DETAIL=DEBUG
#pytest  FlagTree/python/test/tle/unit/test_tle_distributed_d2d.py
#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
#export NCCL_NVLS_ENABLE=0
#export FLAGCX_DMABUF_ENABLE=1
#export FLAGCX_DEBUG=TRACE
#export FLAGCX_DEBUG_SUBSY

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
    echo "ERROR: $CMD failed"
    exit 1
fi
