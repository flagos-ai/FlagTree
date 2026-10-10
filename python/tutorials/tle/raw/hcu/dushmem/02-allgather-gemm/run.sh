#!/bin/bash
# Allgather-GEMM on a DUSHMEM stream.
#   ./run.sh
#   NPES=2 ./run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
NPES="${NPES:-2}"

set +u
if [[ -f /opt/dtk/env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi
set -u

export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/hcu-ag-gemm}"
unset DUSHMEM_BOOTSTRAP

torchrun --nproc_per_node="$NPES" --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29501}" \
  "$ROOT/ag-gemm.py"
