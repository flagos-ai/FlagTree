#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"

NPS=(2 4 8)
ARCH="${ARCH:-sm_90a}"
LAUNCHER="${LAUNCHER:-torchrun}"
TARGET="02-allgather-gemm/ag-gemm.py"

usage() {
    cat <<'EOF'
Usage:
  ./run_ag_gemm_sweep.sh [M_PER_RANK N_PER_RANK K ...]

Runs:
  ./run.sh --launcher torchrun --np {2,4,8} --arch sm_90a 02-allgather-gemm/ag-gemm.py \
    --m-per-rank M --chunk-m M --n-per-rank N --k K --profile --mode check
  then the same command with:
    --mode perf

If no triples are provided, the default shape is:
  1024 1024 1024

Environment overrides:
  ARCH=sm_90a
  LAUNCHER=torchrun
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# == 0 )); then
    SHAPE_VALUES=(
        # m_per_rank n_per_rank k
        1024 1024 1024
        4096 1024 8192
        4096 8192 8192
        4096 28672 8192
        4096 8192 28672
	8192 1024 8192
	8192 8192 8192
	8192 28672 8192
	8192 8192 28672
    )
else
    if (( $# % 3 != 0 )); then
        echo "Shape arguments must be triples: M_PER_RANK N_PER_RANK K" >&2
        usage >&2
        exit 2
    fi
    SHAPE_VALUES=("$@")
fi

on_error() {
    local exit_code=$?
    echo "[failed] command exited with code ${exit_code}; stopping sweep" >&2
    exit "$exit_code"
}
trap on_error ERR

for np in "${NPS[@]}"; do
    for ((i = 0; i < ${#SHAPE_VALUES[@]}; i += 3)); do
        m_per_rank="${SHAPE_VALUES[i]}"
        n_per_rank="${SHAPE_VALUES[i + 1]}"
        k="${SHAPE_VALUES[i + 2]}"

        echo
        echo "[check] np=${np} m_per_rank=${m_per_rank} n_per_rank=${n_per_rank} k=${k}"
        ./run.sh \
            --launcher "$LAUNCHER" \
            --np "$np" \
            --arch "$ARCH" \
            "$TARGET" \
            --m-per-rank "$m_per_rank" \
            --chunk-m "$m_per_rank" \
            --n-per-rank "$n_per_rank" \
            --k "$k" \
            --mode check

	echo
        echo "[perf] np=${np} m_per_rank=${m_per_rank} n_per_rank=${n_per_rank} k=${k}"
        ./run.sh \
            --launcher "$LAUNCHER" \
            --np "$np" \
            --arch "$ARCH" \
            "$TARGET" \
            --m-per-rank "$m_per_rank" \
            --chunk-m "$m_per_rank" \
            --n-per-rank "$n_per_rank" \
            --k "$k" \
            --mode perf
    done
done

echo
echo "[done] ag-gemm sweep completed"
