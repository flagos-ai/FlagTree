#!/usr/bin/env bash

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

# flagtree: this entire file is a flagtree-local rewrite of the upstream version.
# Upstream's script renames OLDER LLVM API names to NEWER ones (because their LLVM
# is newer than cuda-tile v13.3.0's expected LLVM). flagtree's LLVM is OLDER than
# cuda-tile's expected LLVM, so we need the OPPOSITE direction renames, plus a
# couple extra patches that upstream doesn't have.
#
# Patches cuda-tile sources for compatibility with flagtree's pinned LLVM.
#
# flagtree currently pins LLVM commit f6ded0be (v3.6.0 era, ~Jan 2026), which is
# OLDER than the LLVM commit (13c00cbc) that cuda-tile v13.3.0 was developed against.
# This script renames newer LLVM API references in the cuda-tile sources back to
# their older equivalents so the code compiles against flagtree's LLVM.
#
# All patches are idempotent (sed replace; no-op if pattern absent). Safe to re-run.
set -euo pipefail

patch_in_place() {
  local file="$1"; shift
  if [[ ! -f "${file}" ]]; then
    echo "[patch] Target file not found: ${file}" >&2
    return 0
  fi
  if [[ ! -f "${file}.bak" ]]; then
    cp "${file}" "${file}.bak"
  fi
  local tmpfile="${file}.tmp"
  rm -f "${tmpfile}"
  sed "$@" "${file}" > "${tmpfile}" && mv "${tmpfile}" "${file}"
}

ARG_PATH="${1:-${CUDA_TILE_SOURCE_DIR:-}}"
if [[ -z "${ARG_PATH}" ]]; then
  echo "[patch] Base directory not provided and CUDA_TILE_SOURCE_DIR unset" >&2
  exit 1
fi
REPO_ROOT="${ARG_PATH}"
echo "[patch] repo_root=${REPO_ROOT}"

# 1) Global rename: DenseTypedElementsAttr → DenseIntOrFPElementsAttr
# cuda-tile uses the newer LLVM class name DenseTypedElementsAttr (post-rename).
# flagtree's pinned LLVM still uses the older name DenseIntOrFPElementsAttr.
echo "[patch] Global rename: DenseTypedElementsAttr → DenseIntOrFPElementsAttr"
find "${REPO_ROOT}" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) \
  -exec sed -i 's/DenseTypedElementsAttr/DenseIntOrFPElementsAttr/g' {} +

# 2) Rewrite llvm::scope_exit usages for older LLVM
# cuda-tile v13.3.0 uses class-template declaration style:
#   [[maybe_unused]] llvm::scope_exit name(...)
# flagtree's older LLVM only has the factory function llvm::make_scope_exit, so
# rewrite declarations to:
#   [[maybe_unused]] auto name = llvm::make_scope_exit(...)
# Also handle already-partially-patched build-tree copies where the type was
# rewritten to llvm::make_scope_exit but the declaration shape was left intact.
BYTECODE_READER_PATH="${REPO_ROOT}/lib/Bytecode/Reader/BytecodeReader.cpp"
if [[ -f "${BYTECODE_READER_PATH}" ]] && grep -q 'llvm::\(make_\)\?scope_exit' "${BYTECODE_READER_PATH}"; then
  echo "[patch] Rewriting llvm::scope_exit usages in BytecodeReader.cpp"
  patch_in_place "${BYTECODE_READER_PATH}" \
    -e 's/\(\[\[maybe_unused\]\]\) llvm::scope_exit \([A-Za-z_][A-Za-z0-9_]*\)(/\1 auto \2 = llvm::make_scope_exit(/g' \
    -e 's/\(\[\[maybe_unused\]\]\) llvm::make_scope_exit \([A-Za-z_][A-Za-z0-9_]*\)(/\1 auto \2 = llvm::make_scope_exit(/g' \
    -e 's/llvm::scope_exit(/llvm::make_scope_exit(/g'
fi

# 3) Convert isValidRawBuffer 2-arg → 3-arg form
# cuda-tile v13.3.0 calls DenseElementsAttr::isValidRawBuffer(tileType, rawData) — 2-arg.
# flagtree's LLVM only has the 3-arg form: isValidRawBuffer(ShapedType, ArrayRef, bool& detectedSplat).
# Inject a local "bool detectedSplat = false" before the call and pass it as the 3rd argument.
if [[ -f "${BYTECODE_READER_PATH}" ]] && grep -q 'isValidRawBuffer(tileType, rawData))' "${BYTECODE_READER_PATH}"; then
  echo "[patch] Converting isValidRawBuffer 2-arg → 3-arg in BytecodeReader.cpp"
  patch_in_place "${BYTECODE_READER_PATH}" \
    -e 's|// Validate the buffer size and format\.|&\n    bool detectedSplat = false;|' \
    -e 's/isValidRawBuffer(tileType, rawData))/isValidRawBuffer(tileType, rawData, detectedSplat))/g'
fi

echo "[patch] DONE"
