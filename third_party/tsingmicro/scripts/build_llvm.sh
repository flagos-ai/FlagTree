#!/bin/bash
# hash a66376b0dc3b2ea8a84fda26faca287980986f78

if [ -z "${LLVM_PROJECT+x}" ]; then
    echo "Please set the environment variable “LLVM_PROJECT”." 1>&2
    exit 1
fi

if [ ! -d $LLVM_PROJECT ]; then
    echo "Error: $LLVM_PROJECT not exist!" 1>&2
    exit 1
fi

BUILD_TYPE=Release

build_llvm() {
    mkdir $LLVM_PROJECT/build
    cd $LLVM_PROJECT/build
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="clang;mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU;RISCV" \
        -DLLVM_USE_LINKER=lld \
        -DMLIR_ENABLE_BINDINGS_PYTHON=1 \
        -DPython3_EXECUTABLE="$(which python3)" \
        ../llvm
    ninja
}

build_llvm
