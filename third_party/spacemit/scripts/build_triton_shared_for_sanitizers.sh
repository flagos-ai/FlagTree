# using a custom llvm install, builds triton_shared

#!/bin/bash
set -e
set -x

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <existing path to llvm build dir> <existing path to llvm install dir> <existing path to triton shared>"
  exit 1
fi

LLVM_BUILD_PATH="$(realpath "$1")"
LLVM_INSTALL_PATH="$(realpath "$2")"
TRITON_SHARED_PATH="$(realpath "$3")"

# check if the path exists
if [ ! -e "$LLVM_BUILD_PATH" ]; then
  echo "Error: Path '$LLVM_BUILD_PATH' does not exist."
  exit 1
fi

if [ ! -e "$LLVM_INSTALL_PATH" ]; then
  echo "Error: Path '$LLVM_INSTALL_PATH' does not exist."
  exit 1
fi

if [ ! -e "$TRITON_SHARED_PATH" ]; then
  echo "Error: Path '$TRITON_SHARED_PATH' does not exist."
  exit 1
fi

# check whether ~/.triton is empty
# llvm being used from before may be cached and may be different from the custom llvm
# will cause linking errors during the triton-shared build
if [ -e "~/.triton" ]; then
  echo "Error: Please remove ~/.triton and run this script again."
  exit 1
fi

cd "$TRITON_SHARED_PATH"

# prepare for triton_shared build
export PATH="${LLVM_INSTALL_PATH}/bin:${PATH}"
which clang

# build triton-shared with the custom LLVM
cd "${TRITON_SHARED_PATH}/triton"

export TRITON_PLUGIN_DIRS="${TRITON_SHARED_PATH}"
export LLVM_BUILD_DIR="${LLVM_BUILD_PATH}"
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=false python3 -m pip install --no-build-isolation -vvv '.[tests]'

echo "triton-shared build completed"
