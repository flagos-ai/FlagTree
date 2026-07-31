#!/bin/bash
set -euo pipefail
# bash build_whl.sh ${LLVM_INSTALL_DIR} {arch/x86_64/riscv64} {spine-mlir-install-dir} {spine-runtime-install-dir}

# Resolve all input paths to absolute paths to avoid breakage after pushd
LLVM_INSTALL_DIR=$(cd "${1}" && pwd)
BUILD_DIR=build-wheel-${2}
SPINE_MLIR_INSTALL_DIR=$(cd "${3}" && pwd)
chmod a+x "${SPINE_MLIR_INSTALL_DIR}"/bin/*

# spine-runtime install dir (its own `installed/` layout: include/ + lib/).
# libspert.so + spert headers come from here, separate from spine-mlir.
if [ -z "${4:-}" ]; then
    echo "ERROR: SPINE_RUNTIME_INSTALL_DIR (arg 4) is required" >&2
    echo "Usage: bash build_whl.sh <LLVM> <arch> <spine-mlir-install> <spine-runtime-install>" >&2
    exit 1
fi
SPINE_RUNTIME_INSTALL_DIR=$(cd "${4}" && pwd)

CUR_DIR=${PWD}
VERSION_NUMBER=$(cat VERSION_NUMBER)
MAX_JOBS=${MAX_JOBS:-20}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"
echo "SPINE_MLIR_INSTALL_DIR: ${SPINE_MLIR_INSTALL_DIR}"
echo "SPINE_RUNTIME_INSTALL_DIR: ${SPINE_RUNTIME_INSTALL_DIR}"
echo "BUILD_DIR: ${BUILD_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

# Vendor spert headers from SPINE_RUNTIME_INSTALL_DIR/include into backend/include
# (the generated launcher #include "spert.hpp"); libspert.so* is picked up by
# setup.py from SPINE_RUNTIME_INSTALL_DIR/lib.
mkdir -p backend/include
if [ -f "${SPINE_RUNTIME_INSTALL_DIR}/include/spert.hpp" ]; then
    for h in spert.hpp spert_engine.hpp spert_abi.h; do
        [ -f "${SPINE_RUNTIME_INSTALL_DIR}/include/${h}" ] && \
            cp "${SPINE_RUNTIME_INSTALL_DIR}/include/${h}" "backend/include/${h}"
    done
    echo "vendored spert headers from ${SPINE_RUNTIME_INSTALL_DIR}/include"
else
    echo "ERROR: spert.hpp not found in ${SPINE_RUNTIME_INSTALL_DIR}/include" >&2
    exit 1
fi

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git reset
git checkout .
git clean -fd
ls ${CUR_DIR}/patch/*.patch | xargs -n1 git apply

export SPINE_MLIR_INSTALL_DIR=${SPINE_MLIR_INSTALL_DIR}
export SPINE_RUNTIME_INSTALL_DIR=${SPINE_RUNTIME_INSTALL_DIR}
export SPINE_TRITON_VERSION_NUMBER=${VERSION_NUMBER}
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_LIBRARY_DIR=${LLVM_INSTALL_DIR}/lib -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm -DLLD_DIR=${LLVM_INSTALL_DIR}/lib/cmake/lld -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir -DCMAKE_TOOLCHAIN_FILE=${CUR_DIR}/cmake/linux_riscv64.toolchain.cmake"

export CC=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc
export CXX=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false TRITON_IN_TREE_BACKENDS= LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} LLVM_SYSPATH=${LLVM_INSTALL_DIR} MAX_JOBS=${MAX_JOBS} \
python3 setup.py bdist_wheel --plat=linux-${2}

cp dist/*.whl ${CUR_DIR}/${BUILD_DIR}/
popd

echo "whl package generated successfully at: ${CUR_DIR}/${BUILD_DIR}/"
