#!/bin/bash

set -e

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi
echo "${realpath}"
TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64
TRITON=$project_dir

if [ ! -d $TX8_DEPS_ROOT ] || [ ! -d $LLVM ]; then
    WORKSPACE="${HOME}/.triton/tsingmicro/"
    TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
    LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64
fi

if [ ! -d $TX8_DEPS_ROOT ]; then
    echo "Error: $TX8_DEPS_ROOT not exist!" 1>&2
    exit 1
fi

if [ ! -d $LLVM ]; then
    echo "Error: $LLVM not exist!" 1>&2
    exit 1
fi

# Default values
BUILD_TYPE="release"
ACTION="install"
SCRIPT_NAME=$(basename "$0")

# Function to display usage
usage() {
    echo "Usage: $SCRIPT_NAME [-t build_type] [-a action]"
    echo "Options:"
    echo "  -t build_type  Specify build type: debug or release (default: release)"
    echo "  -a action      Specify action: install or wheel (default: install)"
    echo "  -h             Display this help message"
    exit 1
}

# Process command-line options with getopts
while getopts ":t:a:h" opt; do
    case $opt in
        t)
            BUILD_TYPE="$OPTARG"
            # Convert to lowercase for case-insensitive comparison
            BUILD_TYPE=$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')
            # Validate build_type
            if [[ "$BUILD_TYPE" != "debug" && "$BUILD_TYPE" != "release" ]]; then
                echo "Error: Invalid build type '$BUILD_TYPE'. Must be 'debug' or 'release'." >&2
                usage
            fi
            ;;
        a)
            ACTION="$OPTARG"
            ACTION=$(echo "$ACTION" | tr '[:upper:]' '[:lower:]')
            # Validate action
            if [[ "$ACTION" != "install" && "$ACTION" != "wheel" ]]; then
                echo "Error: Invalid action '$ACTION'. Must be 'install' or 'wheel'." >&2
                usage
            fi
            ;;
        h)
            usage
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

echo "Build configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Action: $ACTION"

build_triton() {
    if [ "$BUILD_TYPE" == "debug" ]; then
        export DEBUG=ON
    else
        export REL_WITH_DBG_INFO=ON
    fi

    export TRITON_BUILD_WITH_CLANG_LLD=true
    export TRITON_BUILD_WITH_CCACHE=true
    export TRITON_OFFLINE_BUILD=OFF
    export TRITON_BUILD_PROTON=OFF
    export CXXFLAGS="-Wno-dangling-assignment-gsl"

    echo "export TRITON_OFFLINE_BUILD=$TRITON_OFFLINE_BUILD"
    echo "export TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD"
    echo "export TRITON_BUILD_WITH_CCACHE=$TRITON_BUILD_WITH_CCACHE"
    echo "export TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON"

    cd $TRITON/python
    build_opt=install

    if [ "$ACTION" == "wheel" ]; then
        build_opt=wheel
    fi

    python3 -m pip $build_opt . --no-build-isolation -v --verbose
}

if [ -f $TRITON/.venv/bin/activate ]; then
    source $TRITON/.venv/bin/activate
fi

export LLVM_SYSPATH=$LLVM
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export FLAGTREE_BACKEND=tsingmicro

# debug
# export USE_HOST_PROFILE=1
# export NO_INTRNISIC_RUN=1

echo "export TX8_DEPS_ROOT=$TX8_DEPS_ROOT"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"

build_triton
