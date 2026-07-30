# !/bin/bash
# bash build.sh ${LLVM_INSTALL_DIR} {arch/x86_64/riscv64} {spine-mlir-install-dir}

# Resolve all input paths to absolute paths to avoid breakage after pushd
LLVM_INSTALL_DIR=$(cd "${1}" && pwd)
BUILD_DIR=build-${2}
SPINE_MLIR_INSTALL_DIR=$(cd "${3}" && pwd)
chmod a+x "${SPINE_MLIR_INSTALL_DIR}"/bin/*


CUR_DIR=${PWD}
VERSION_NUMBER=$(cat VERSION_NUMBER)
MAX_JOBS=${MAX_JOBS:-20}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"
echo "SPINE_MLIR_INSTALL_DIR: ${SPINE_MLIR_INSTALL_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git reset
git checkout .
git clean -fd
ls ${CUR_DIR}/patch/*.patch | xargs -n1 git apply

export SPINE_MLIR_INSTALL_DIR=${SPINE_MLIR_INSTALL_DIR}
export SPINE_TRITON_VERSION_NUMBER=${VERSION_NUMBER}
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_LIBRARY_DIR=${LLVM_INSTALL_DIR}/lib -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm -DLLD_DIR=${LLVM_INSTALL_DIR}/lib/cmake/lld -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir -DCMAKE_TOOLCHAIN_FILE=${CUR_DIR}/cmake/linux_riscv64.toolchain.cmake"

export CC=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc
export CXX=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false TRITON_IN_TREE_BACKENDS= LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} LLVM_SYSPATH=${LLVM_INSTALL_DIR} MAX_JOBS=${MAX_JOBS} \
python3 setup.py install --prefix=${TRITON_PLUGIN_DIRS}/${BUILD_DIR}
popd

rm -rf ${BUILD_DIR}/triton

if ls -d ${BUILD_DIR}/lib/python*/site-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/lib/python*/site-packages/triton* ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/lib
elif  ls -d ${BUILD_DIR}/local/lib/python*/dist-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/local/lib/python*/dist-packages/triton* ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/local
else
    echo "Error: Cannot find triton package"
    exit 1
fi
