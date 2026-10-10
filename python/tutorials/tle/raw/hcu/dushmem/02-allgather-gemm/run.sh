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

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export DUSHMEM_BOOTSTRAP=MPI
export OMPI_MCA_coll='^hcoll'
export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/hcu-ag-gemm}"

mpirun --allow-run-as-root -n "$NPES" python3 "$ROOT/ag-gemm.py"
