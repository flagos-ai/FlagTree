#!/bin/bash
# For multi-node runs, set NNODES, NODE_RANK, MASTER_ADDR and MASTER_PORT.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/flagcx_env.sh" "${1:-}"

export FLAGCX_P2P_DISABLE=1

for path in "${xpu_home}" "${xccl_home}"; do
    if [ ! -d "${path}" ]; then
        echo "ERROR: SDK directory not found: ${path}" >&2
        echo "Set XPU_HOME or XCCL_HOME to override the defaults in ${sdk_dir}" >&2
        exit 1
    fi
done

run_test "${script_dir}/test_tle_signal_wait.py" \
    "${NPROC_PER_NODE:-2}" "${NNODES:-1}" "${NODE_RANK:-0}" \
    "${MASTER_ADDR:-}" "${MASTER_PORT:-}"
