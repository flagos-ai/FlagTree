#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpu_home="${XPU_HOME:-${DEVICE_HOME:-/usr/local/xpu}}"
xccl_home="${XCCL_HOME:-${CCL_HOME:-/usr/local/xccl}}"
export PATH="${xpu_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${xpu_home}/so:${xpu_home}/lib64:${xccl_home}/so${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FLAGCX_XSHMEM_DEVICE_LIB="${FLAGCX_XSHMEM_DEVICE_LIB:-${xccl_home}/lib/libxshmem_device.a}"
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"
export TRITON_STORE_BINARY_ONLY=0

# The workers load the allocator before the process group is initialized.
LOCAL_RANK=0 python - <<'PY'
from pathlib import Path
from tempfile import gettempdir

allocator = Path(gettempdir()) / "flagcx_allocator.so"
if not allocator.is_file():
    from triton.experimental.tle.language.communication import compile_flagcx_allocator

    compile_flagcx_allocator()
    if not allocator.is_file():
        raise SystemExit(f"Failed to build {allocator}")
PY

exec torchrun --standalone --nnodes=1 --nproc_per_node=2 "${script_dir}/test_tle_get_local_pe.py"
