#!/usr/bin/env bash
set -euo pipefail

lock_file="${TRITON_TLE_NVCC_LOCK:-/tmp/triton_tle_nvcc.lock}"
real_nvcc="${REAL_NVCC:-/usr/local/cuda-12.8/bin/nvcc}"

exec 9>"${lock_file}"
flock 9
exec "${real_nvcc}" "$@"
