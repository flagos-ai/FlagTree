#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
NVSHMRUN_BIN="${NVSHMRUN:-nvshmrun}"
NP=2
ARCH=""
FORCE=0
NVSHMEM_HOME_ARG="${NVSHMEM_HOME:-}"

usage() {
    cat <<'EOF'
Usage:
  run_nvshmem.sh [options] <example.py> [-- <example arguments...>]

Options:
  --np N                     Number of NVSHMEM processes (default: 2)
  --python PATH              Python executable (default: $PYTHON or python)
  --nvshmrun PATH            nvshmrun executable (default: $NVSHMRUN or nvshmrun)
  --nvshmem-home PATH        NVSHMEM installation root (or set NVSHMEM_HOME)
  --arch ARCH                CUDA architecture (default: sm_80)
  --force                    Rebuild the host shared library
  -h, --help                 Show this help

Examples:
  ./run_nvshmem.sh --np 2 02-ring-reduce/ring-reduce.py
  ./run_nvshmem.sh --np 2 06-ring-reduce-overlap/overlap.py
  ./run_nvshmem.sh --np 4 ring-reduce.py
EOF
}

while (($#)); do
    case "$1" in
        --np)
            NP="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --nvshmrun)
            NVSHMRUN_BIN="$2"
            shift 2
            ;;
        --nvshmem-home)
            NVSHMEM_HOME_ARG="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            TARGET_INPUT="$1"
            shift
            break
            ;;
    esac
done

if [[ -z "${TARGET_INPUT:-}" ]]; then
    usage >&2
    exit 2
fi

if [[ ! "$NP" =~ ^[1-9][0-9]*$ ]]; then
    echo "Process count must be a positive integer: $NP" >&2
    exit 2
fi

if [[ -z "$NVSHMEM_HOME_ARG" ]]; then
    echo "NVSHMEM_HOME is required; set it or pass --nvshmem-home PATH" >&2
    exit 2
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi

if [[ -f "$TARGET_INPUT" ]]; then
    TARGET="$(cd -- "$(dirname -- "$TARGET_INPUT")" && pwd)/$(basename -- "$TARGET_INPUT")"
elif [[ -f "$SCRIPT_DIR/$TARGET_INPUT" ]]; then
    TARGET="$SCRIPT_DIR/$TARGET_INPUT"
else
    mapfile -t MATCHES < <(find "$SCRIPT_DIR" -mindepth 2 -maxdepth 2 -type f -name "$TARGET_INPUT")
    if ((${#MATCHES[@]} != 1)); then
        echo "Could not uniquely resolve example: $TARGET_INPUT" >&2
        exit 2
    fi
    TARGET="${MATCHES[0]}"
fi

if ! NVSHMRUN_PATH="$(command -v "$NVSHMRUN_BIN")"; then
    echo "Could not find nvshmrun executable: $NVSHMRUN_BIN" >&2
    exit 2
fi
NVSHMRUN_BIN="$(cd -- "$(dirname -- "$NVSHMRUN_PATH")" && pwd)/$(basename -- "$NVSHMRUN_PATH")"

if [[ -n "$NVSHMEM_HOME_ARG" ]]; then
    NVSHMEM_HOME_ARG="$(cd -- "$NVSHMEM_HOME_ARG" && pwd)"
fi

PREPARE_ARGS=(
    "$SCRIPT_DIR/common/prepare_nvshmem.py"
    "$TARGET"
    "--nvshmem-home"
    "$NVSHMEM_HOME_ARG"
)
[[ -n "$ARCH" ]] && PREPARE_ARGS+=("--arch" "$ARCH")
((FORCE)) && PREPARE_ARGS+=("--force")

"$PYTHON_BIN" "${PREPARE_ARGS[@]}"

export NVSHMEM_HOME="$NVSHMEM_HOME_ARG"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd -- "$(dirname -- "$TARGET")"
exec "$NVSHMRUN_BIN" -np "$NP" "$PYTHON_BIN" "$(basename -- "$TARGET")" "$@"
