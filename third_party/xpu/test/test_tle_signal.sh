#!/usr/bin/env bash
# Runs test_tle_signal.py on KunlunXin XPU. Before running this
# script, it is necessary to set XPU_HOME and XCCL_HOME, and the environment
# variables NNODES, NODE_RANK, MASTER_ADDR and MASTER_PORT for the distributed
# nodes. Set CUDA_VISIBLE_DEVICES if the first NPROC_PER_NODE cards are not the
# ones to use.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
flagtree_root="$(cd "${script_dir}/../../.." && pwd)"
test_py="${script_dir}/test_tle_signal.py"

# XPU SDK and XCCL, under the same variable names the FlagTree XPU install reads.
xpu_home="${XPU_HOME:-${DEVICE_HOME:-/usr/local/xpu}}"
xccl_home="${XCCL_HOME:-${CCL_HOME:-/usr/local/xccl}}"
export PATH="${xpu_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${xpu_home}/so:${xpu_home}/lib64:${xccl_home}/so${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# make_xpubin pre-links the vendor XSHMEM device library into the kernel. The
# backend derives it from XCCL_HOME, so only set this when it sits elsewhere.
export FLAGCX_XSHMEM_DEVICE_LIB="${FLAGCX_XSHMEM_DEVICE_LIB:-${xccl_home}/lib/libxshmem_device.a}"

export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=1

nproc_per_node="${NPROC_PER_NODE:-2}"
nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
master_port="${MASTER_PORT:-8352}"

if [[ "${nnodes}" -gt 1 ]]; then
    master_addr="${MASTER_ADDR:?The multi-node runtime must set MASTER_ADDR (the reachable IP or hostname of node 0)}"
else
    master_addr="${MASTER_ADDR:-localhost}"
fi

if [[ "${node_rank}" -lt 0 || "${node_rank}" -ge "${nnodes}" ]]; then
    echo "NODE_RANK=${node_rank} is out of bounds. Valid range is 0 to $((nnodes - 1))" >&2
    exit 1
fi

for path in "${xpu_home}" "${xccl_home}" "${FLAGCX_XSHMEM_DEVICE_LIB}" "${test_py}"; do
    if [[ ! -e "${path}" ]]; then
        echo "not found: ${path}" >&2
        echo "set XPU_HOME / XCCL_HOME / FLAGCX_XSHMEM_DEVICE_LIB to the XPU install" >&2
        exit 1
    fi
done

echo "Starting TLE signal test: node_rank=${node_rank}/${nnodes}, "
echo "nproc_per_node=${nproc_per_node}, master=${master_addr}:${master_port}"

exec torchrun \
    --nproc_per_node="${nproc_per_node}" \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    "${test_py}"
