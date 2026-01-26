#!/bin/bash

function print_usage() {
    echo "Usage: source build_ttenv.sh <llvm_install_dir>"
}

# 确保这个脚本由 source 命令执行
if [[ "$0" == "${BASH_SOURCE[0]}" ]]; then
    echo "This script must be executed by 'source' or '.' command."
    exit 1
fi

if [ $# -ne 1 ] || [ ! -d $1 ] ; then
    print_usage
    return 1
fi

# 需要在triton的根目录执行这个脚本
if [ ! -d 'python/triton' ] || [ ! -d 'third_party/sunrise' ] ; then
    echo "This script must be executed in triton project root directory!"
    return 1
fi

LLVM_INSTALL_DIR=$1
export LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib
export LLVM_SYSPATH=$LLVM_INSTALL_DIR
export MLIR_DIR=$LLVM_LIBRARY_DIR/cmake/mlir

export TRITON_OFFLINE_BUILD=1
export TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_WITH_CLANG_LLD=1
export MAX_JOBS=50
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tangrt/lib/linux-x86_64/stub

export FLAGTREE_BACKEND=sunrise

# 拷贝install_dir中缺失的bitcode和Filecheck

mkdir -p third_party/sunrise/backend/lib
cp $1/stpu/bitcode/*.bc third_party/sunrise/backend/lib
if [ $? -ne 0 ] ; then
    echo "copy stpu bitcode failed."
    return 1
fi

# 必须有libtang.so.0.19.2这个文件
if [ ! -f /usr/local/tangrt/lib/linux-x86_64/stub/libtang.so.0.19.2 ] ; then
    ln -s /usr/local/tangrt/lib/linux-x86_64/stub/libtang.so /usr/local/tangrt/lib/linux-x86_64/stub/libtang.so.0.19.2
fi

echo "--- OK ---"
