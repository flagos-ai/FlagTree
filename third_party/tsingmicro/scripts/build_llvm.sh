#!/bin/bash
# Build LLVM/MLIR and produce a portable tarball.
#
#   LLVM_PROJECT=/path/to/llvm-project ./build_llvm.sh
#
# Products next to the source tree:
#   $LLVM_PROJECT/build/                build tree
#   $LLVM_PROJECT/install/              install prefix (override: LLVM_INSTALL_PREFIX)
#   $LLVM_PROJECT/llvm-<major>.tar.gz   portable package (absolute symlinks materialized)
#
# hash 1f126a6dea50d185c0781743a667390037ae88bd

set -euo pipefail

if [ -z "${LLVM_PROJECT:-}" ]; then
    echo "error: LLVM_PROJECT is not set" >&2
    exit 1
fi
[ -d "$LLVM_PROJECT" ] || { echo "error: $LLVM_PROJECT does not exist" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/env.sh
source "$script_dir/lib/env.sh"

BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_PROJECT/build}"
INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-$LLVM_PROJECT/install}"

# Keep the previous install tree instead of destroying a 30-60 min artifact.
if [ -d "$INSTALL_PREFIX" ]; then
    backup="$INSTALL_PREFIX.old.$(date +%Y%m%d_%H%M%S)"
    echo "[llvm] keeping the previous prefix as $backup"
    mv "$INSTALL_PREFIX" "$backup"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="clang;mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU;RISCV" \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=1 \
    -DPython3_EXECUTABLE="$PYTHON" \
    ../llvm

ninja
cmake --install . --prefix "$INSTALL_PREFIX"

major="$("$INSTALL_PREFIX/bin/llvm-config" --version 2>/dev/null | cut -d. -f1)"
[ -n "$major" ] || { echo "error: cannot determine the LLVM major version from $INSTALL_PREFIX" >&2; exit 1; }

tsm_package_llvm "$INSTALL_PREFIX" "$LLVM_PROJECT/llvm-$major.tar.gz" "llvm-$major"
