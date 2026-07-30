#!/bin/bash
# Build spine-triton for x86 (native) with riscv64 target support
# Usage: bash build_x86.sh ${LLVM_INSTALL_DIR} ${SPINE_MLIR_INSTALL_DIR}
set -e

LLVM_INSTALL_DIR=$(cd "${1}" && pwd)
SPINE_MLIR_INSTALL_DIR=$(cd "${2}" && pwd)
chmod a+x "${SPINE_MLIR_INSTALL_DIR}"/bin/*

CUR_DIR=${PWD}
VERSION_NUMBER=$(cat VERSION_NUMBER)
MAX_JOBS=${MAX_JOBS:-20}

echo "=== Building spine-triton for x86 ==="
echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"
echo "SPINE_MLIR_INSTALL_DIR: ${SPINE_MLIR_INSTALL_DIR}"

BUILD_DIR=build-x86_64
export TRITON_PLUGIN_DIRS=${PWD}

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git reset 2>/dev/null || true
git checkout . 2>/dev/null || true
git clean -fd 2>/dev/null || true
for p in ${CUR_DIR}/patch/*.patch; do
    echo "Applying patch: ${p}"
    git apply "${p}" || { echo "ERROR: failed to apply patch ${p}"; exit 1; }
done

export SPINE_MLIR_INSTALL_DIR=${SPINE_MLIR_INSTALL_DIR}
export SPINE_TRITON_VERSION_NUMBER=${VERSION_NUMBER}
# No toolchain file for x86 native build
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_LIBRARY_DIR=${LLVM_INSTALL_DIR}/lib -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm -DLLD_DIR=${LLVM_INSTALL_DIR}/lib/cmake/lld -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir"

unset CC
unset CXX

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false TRITON_IN_TREE_BACKENDS= LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} LLVM_SYSPATH=${LLVM_INSTALL_DIR} MAX_JOBS=${MAX_JOBS} \
python3 setup.py install --prefix=${TRITON_PLUGIN_DIRS}/${BUILD_DIR}
popd

rm -rf ${BUILD_DIR}/triton

if ls -d ${BUILD_DIR}/lib/python*/site-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/lib/python*/site-packages/triton* ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/lib
elif ls -d ${BUILD_DIR}/local/lib/python*/dist-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/local/lib/python*/dist-packages/triton* ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/local
else
    echo "Error: Cannot find triton package"
    exit 1
fi

echo "=== Build complete ==="
echo "Output: ${CUR_DIR}/${BUILD_DIR}/"
