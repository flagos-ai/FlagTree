# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
