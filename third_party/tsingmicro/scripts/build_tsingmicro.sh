#!/bin/bash

set -e

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

# shellcheck source=lib/env.sh
source "$script_dir/lib/env.sh"

# Default values
BUILD_TYPE="release"
ACTION="wheel"
SCRIPT_NAME=$(basename "$0")

# Function to display usage
usage() {
    echo "Usage: $SCRIPT_NAME [-t build_type] [-a action]"
    echo "Options:"
    echo "  -t build_type  Specify build type: debug or release (default: release)"
    echo "  -a action      Specify action: install or wheel (default: wheel)"
    echo "  -h             Display this help message"
    exit "${1:-1}"
}

# Process command-line options with getopts
while getopts ":t:a:h" opt; do
    case $opt in
        t)
            BUILD_TYPE="${OPTARG,,}"
            # Validate build_type
            if [[ "$BUILD_TYPE" != "debug" && "$BUILD_TYPE" != "release" ]]; then
                echo "Error: Invalid build type '$BUILD_TYPE'. Must be 'debug' or 'release'." >&2
                usage
            fi
            ;;
        a)
            ACTION="${OPTARG,,}"
            # Validate action
            if [[ "$ACTION" != "install" && "$ACTION" != "wheel" ]]; then
                echo "Error: Invalid action '$ACTION'. Must be 'install' or 'wheel'." >&2
                usage
            fi
            ;;
        h)
            usage 0
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Shift off the options and optional arguments
shift $((OPTIND - 1))
[ $# -eq 0 ] || { echo "error: unexpected argument(s): $*" >&2; usage; }
tsm_env_check || exit 1

echo "Build configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Action: $ACTION"

die() { echo "error: $*" >&2; exit 1; }


build_triton() {

    if [ "$BUILD_TYPE" == "debug" ]; then
        export DEBUG=ON
    else
        export REL_WITH_DBG_INFO=ON
    fi

    export TRITON_BUILD_WITH_CLANG_LLD=true
    export TRITON_BUILD_WITH_CCACHE=true
    export TRITON_OFFLINE_BUILD="${TRITON_OFFLINE_BUILD:-ON}"
    export TRITON_BUILD_PROTON="${TRITON_BUILD_PROTON:-OFF}"

    echo "export TRITON_OFFLINE_BUILD=$TRITON_OFFLINE_BUILD"
    echo "export TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD"
    echo "export TRITON_BUILD_WITH_CCACHE=$TRITON_BUILD_WITH_CCACHE"
    echo "export TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON"

    cd "$TRITON" || die "cannot enter $TRITON"

    if [ "$ACTION" == "wheel" ]; then
        local out wheel max
        out=$(mktemp -d) || die "cannot create a temporary directory"
        "$PYTHON" -m pip wheel . --no-build-isolation -v --verbose -w "$out"
        wheel=$(ls "$out"/*.whl 2>/dev/null | head -1)
        [ -n "$wheel" ] || die "wheel not found in $out"
        max=$(tsm_wheel_glibc_max "$wheel") || max=""
        [ -n "$max" ] || die "cannot read the glibc requirement of $wheel"
        cp -f "$wheel" "$TRITON/"
        echo "[abi] wheel glibc_max=$max ($(basename "$wheel"))"
        rm -rf "$out"
    else
        "$PYTHON" -m pip install . --no-build-isolation -v --verbose
    fi
}

tsm_activate_venv
export FLAGTREE_BACKEND=tsingmicro

# debug
# export NO_INTRNISIC_RUN=1

tsm_dump_env TX8_DEPS_ROOT LLVM_SYSPATH FLAGTREE_BACKEND

# synchronous temporary solution: add waitfinish after every cintrinsic exec
export ENABLE_SYNCHRONOUS_INTRINSIC=0
tsm_dump_env ENABLE_SYNCHRONOUS_INTRINSIC

build_triton
