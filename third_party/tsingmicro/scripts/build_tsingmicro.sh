#!/bin/bash

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64

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

BUILD_TYPE=Release

build_wheel=OFF
arg1=$1

if [ "$arg1" = "wheel" ]; then
    build_wheel=ON
    echo "build wheel"
fi

build_triton() {
    if [ "x$BUILD_TYPE" == "xDebug" ]; then
        export DEBUG=ON
    else
        export REL_WITH_DBG_INFO=ON
    fi

    export TRITON_BUILD_WITH_CLANG_LLD=true
    export TRITON_BUILD_WITH_CCACHE=true
    export TRITON_OFFLINE_BUILD=ON
    export TRITON_BUILD_PROTON=OFF

    echo "export TRITON_OFFLINE_BUILD=$TRITON_OFFLINE_BUILD"
    echo "export TRITON_BUILD_WITH_CLANG_LLD=$TRITON_BUILD_WITH_CLANG_LLD"
    echo "export TRITON_BUILD_WITH_CCACHE=$TRITON_BUILD_WITH_CCACHE"
    echo "export TRITON_BUILD_PROTON=$TRITON_BUILD_PROTON"

    cd $project_dir/python
    build_opt=install

    if [ "x$build_wheel" == "xON" ]; then
        build_opt=wheel
    fi
    python3 -m pip $build_opt . --no-build-isolation -v --verbose
}

if [ -f $project_dir/.venv/bin/activate ]; then
    source $project_dir/.venv/bin/activate
fi

export LLVM_SYSPATH=$LLVM
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export FLAGTREE_BACKEND=tsingmicro

echo "export TX8_DEPS_ROOT=$TX8_DEPS_ROOT"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"

build_triton
