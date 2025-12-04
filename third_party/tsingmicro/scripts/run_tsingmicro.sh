#!/bin/bash

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

TRITON=$WORKSPACE/triton
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

if [ -f $TRITON/.venv/bin/activate ]; then
    source $TRITON/.venv/bin/activate
fi

# 必须的
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export LLVM_SYSPATH=$LLVM

# 后续需要优化删除的
export PYTHONPATH=$LLVM/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
export VENDOR_VERSION=1

# 非必须的 调试相关
export TRITON_DUMP_PATH=$TRITON/dump
export TRITON_ALWAYS_COMPILE=1

echo "export TX8_DEPS_ROOT=$TX8_DEPS_ROOT"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"
echo "export PYTHONPATH=$PYTHONPATH"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "export VENDOR_VERSION=$VENDOR_VERSION"
echo "export TRITON_DUMP_PATH=$TRITON_DUMP_PATH"
echo "export TRITON_ALWAYS_COMPILE=$TRITON_ALWAYS_COMPILE"

USE_SIM_MODE=${USE_SIM_MODE} python3 $@
