# build a custom llvm for sanitizers
# this build uses the same version of LLVM that is currently supported by spine-triton
# and installs the required projects that the sanitizers need (compiler-rt, clang)
# users need to activate venv before running this script

#!/bin/bash
set -e
set -x

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <desired path to LLVM installation directory> <existing path to spine triton>"
  exit 1
fi

LLVM_PATH="$(realpath "$1")"
SPINE_TRITON_PATH="$(realpath "$2")"

# check if the path exists
if [ ! -e "$LLVM_PATH" ]; then
  echo "Error: Path '$LLVM_PATH' does not exist."
  exit 1
fi

if [ ! -e "$SPINE_TRITON_PATH" ]; then
  echo "Error: Path '$SPINE_TRITON_PATH' does not exist."
  exit 1
fi

echo "Installing LLVM to path: $LLVM_PATH"
cd $LLVM_PATH

LLVM_HASH_FILE="${SPINE_TRITON_PATH}/triton/cmake/llvm-hash.txt"
if [ ! -e "${LLVM_HASH_FILE}" ]; then
  print_error "${LLVM_HASH_FILE} does not exist"
fi

LLVM_BUILD_DIR="${LLVM_PATH}/llvm-build"
LLVM_INSTALL_DIR="${LLVM_PATH}/llvm-install"
LLVM_SOURCE_DIR="${LLVM_PATH}/llvm-project"
LLVM_SOURCE="${LLVM_SOURCE_DIR}/llvm"

# compiler-rt and clang are the sanitizer-specific LLVM projects
# openmp is used for parallelizing the triton grid for ThreadSanitizer (TSan)
LLVM_PROJECTS="clang;compiler-rt;openmp;mlir"

# these are the targets supported by the Triton language
# Triton's build script for LLVM uses these exact targets
# see https://github.com/triton-lang/triton/blob/main/scripts/build-llvm-project.sh
LLVM_TARGETS="Native;NVPTX;AMDGPU"

mkdir -p "${LLVM_BUILD_DIR}"
mkdir -p "${LLVM_INSTALL_DIR}"

git clone https://github.com/llvm/llvm-project "${LLVM_SOURCE_DIR}"

LLVM_HASH=$(cat "${LLVM_HASH_FILE}")

cd "${LLVM_SOURCE_DIR}"
git checkout ${LLVM_HASH}

export CXXFLAGS="-Wno-unused-command-line-argument $CXXFLAGS"
export CFLAGS="-Wno-unused-command-line-argument $CFLAGS"

cd "$LLVM_BUILD_DIR"
cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang               \
  -DCMAKE_CXX_COMPILER=clang++           \
  -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR    \
  -DLLVM_ENABLE_PROJECTS=$LLVM_PROJECTS       \
  -DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS \
  $LLVM_SOURCE

echo "Installing LLVM to: $LLVM_INSTALL_DIR"
ninja -j$(nproc --all) -l$(nproc --all) || exit -1
ninja -j$(nproc --all) -l$(nproc --all) install

echo "LLVM installation succeeded"
