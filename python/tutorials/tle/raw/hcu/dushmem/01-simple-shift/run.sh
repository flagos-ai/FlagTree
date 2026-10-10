#!/bin/bash
# Launch the DUSHMEM simple_shift example.
#   ./run.sh        raw kernel via torchrun, 2 PEs
#   ./run.sh pure   the manual HIP source
#   ./run.sh perf   raw versus pure HIP
#   NPES=4 ./run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
NPES="${NPES:-2}"
MODE="${1:-run}"

set +u
if [[ -f /opt/dtk/env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi
set -u

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_coll='^hcoll'
export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
unset DUSHMEM_BOOTSTRAP

case "$MODE" in
  run)
    torchrun --nproc_per_node="$NPES" --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
      "$ROOT/simple-shift.py"
    ;;
  pure)
    export DUSHMEM_BOOTSTRAP=MPI
    hipcc -fgpu-rdc --offload-arch=gfx936 -O3 \
      -DHIP_ENABLE_WARP_SYNC_BUILTINS -mcode-object-version=4 \
      -I/opt/dtk/include -I/opt/dtk/include/dushmem \
      -L/opt/dtk/lib/dushmem \
      "$ROOT/simple-shift-pure.hip" -o /tmp/dushmem-simple-shift \
      -ldushmem_host -ldushmem_device
    mpirun --allow-run-as-root -n "$NPES" /tmp/dushmem-simple-shift
    ;;
  perf)
    python3 "$ROOT/perf.py"
    ;;
  *)
    echo "usage: $0 [run|pure|perf]" >&2
    exit 1
    ;;
esac
