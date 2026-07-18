#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
NVSHMRUN_BIN="${NVSHMRUN:-nvshmrun}"
TORCHRUN_BIN="${TORCHRUN:-torchrun}"
LAUNCHER="${LAUNCHER:-nvshmrun}"
NP=2
ARCH="sm_90"
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
  --arch ARCH                CUDA architecture (default: sm_90)
  --force                    Rebuild the host shared library
  -h, --help                 Show this help

Examples:
  ./run.sh --np 2 01-simple-shift/simple-shift.py
EOF
}

while (($#)); do
    case "$1" in
        --np)
            NP="$2"
            shift 2
            ;;
        --launcher)
            LAUNCHER="$2"
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

if [[ "$LAUNCHER" != "nvshmrun" && "$LAUNCHER" != "torchrun" ]]; then
    echo "Launcher must be one of: nvshmrun, torchrun" >&2
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

if [[ "$LAUNCHER" == "nvshmrun" ]]; then
    if ! NVSHMRUN_PATH="$(command -v "$NVSHMRUN_BIN")"; then
        echo "Could not find nvshmrun executable: $NVSHMRUN_BIN" >&2
        exit 2
    fi
    NVSHMRUN_BIN="$(cd -- "$(dirname -- "$NVSHMRUN_PATH")" && pwd)/$(basename -- "$NVSHMRUN_PATH")"
else
    if ! TORCHRUN_PATH="$(command -v "$TORCHRUN_BIN")"; then
        echo "Could not find torchrun executable: $TORCHRUN_BIN" >&2
        exit 2
    fi
    TORCHRUN_BIN="$(cd -- "$(dirname -- "$TORCHRUN_PATH")" && pwd)/$(basename -- "$TORCHRUN_PATH")"
fi

if [[ -n "$NVSHMEM_HOME_ARG" ]]; then
    NVSHMEM_HOME_ARG="$(cd -- "$NVSHMEM_HOME_ARG" && pwd)"
fi

PREPARE_ARGS=(
    "$SCRIPT_DIR/common/build.py"
    "$TARGET"
    "--nvshmem-home"
    "$NVSHMEM_HOME_ARG"
)
[[ -n "$ARCH" ]] && PREPARE_ARGS+=("--arch" "$ARCH")
((FORCE)) && PREPARE_ARGS+=("--force")

"$PYTHON_BIN" "${PREPARE_ARGS[@]}"

export NVSHMEM_HOME="$NVSHMEM_HOME_ARG"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd -- "$(dirname -- "$TARGET")"
if [[ "$LAUNCHER" == "nvshmrun" ]]; then
    exec "$NVSHMRUN_BIN" -np "$NP" "$PYTHON_BIN" "$(basename -- "$TARGET")" "$@"
else
    exec "$TORCHRUN_BIN" --nproc_per_node="$NP" "$(basename -- "$TARGET")" "$@"
fi
