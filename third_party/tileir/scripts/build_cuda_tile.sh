
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Usage: $0 <cuda_tile_repo_root>" >&2
  exit 2
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Repo root does not exist: ${REPO_ROOT}" >&2
  exit 2
fi

: "${LLVM_SYSPATH:?LLVM_SYSPATH is required}"
LLVM_EXTERNAL_LIT="${LLVM_EXTERNAL_LIT:-${LLVM_SYSPATH}/bin/llvm-lit}"

BUILD_DIR="${REPO_ROOT}/build"
INSTALL_DIR="${REPO_ROOT}/build/install"
JOBS="${NINJA_JOBS:-32}"

# Clean previous build and install results
rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCUDA_TILE_USE_LLVM_INSTALL_DIR="${LLVM_SYSPATH}" \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

cmake --build "${BUILD_DIR}" --target install -- -j"${JOBS}"
