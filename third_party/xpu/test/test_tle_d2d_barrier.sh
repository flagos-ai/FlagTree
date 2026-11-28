#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/flagcx_env.sh" "${1:-}"

export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"
export TRITON_STORE_BINARY_ONLY=0

run_test "${script_dir}/test_tle_d2d_barrier.py"
