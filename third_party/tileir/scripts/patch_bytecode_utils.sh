#!/usr/bin/env bash
set -euo pipefail

patch_in_place() {
  local file="$1"; shift
  if [[ ! -f "${file}" ]]; then
    echo "[patch] Target file not found: ${file}" >&2
    exit 1
  fi

  if [[ ! -f "${file}.bak" ]]; then
    cp "${file}" "${file}.bak"
  fi

  local tmpfile="${file}.tmp"
  rm -f "${tmpfile}"

  # Keep each sed argument intact (some expressions include spaces).
  sed "$@" "${file}" > "${tmpfile}"
  if cmp -s "${file}" "${tmpfile}"; then
    rm "${tmpfile}"
  else
    mv "${tmpfile}" "${file}"
  fi
}

# Treat the argument as the extracted cuda_tile repo root (preferred), or fall back
# to CUDA_TILE_SOURCE_DIR (used by CMake).
ARG_PATH="${1:-${CUDA_TILE_SOURCE_DIR:-}}"
if [[ -z "${ARG_PATH}" ]]; then
  echo "[patch] Base directory not provided and CUDA_TILE_SOURCE_DIR unset" >&2
  exit 1
fi

# Allow passing either repo root or a direct file path (legacy behavior).
if [[ "${ARG_PATH}" == *.cpp || "${ARG_PATH}" == *.td ]]; then
  REPO_ROOT="$(cd "$(dirname "${ARG_PATH}")/.." && pwd)"
else
  REPO_ROOT="${ARG_PATH}"
fi

BYTECODE_UTIL_PATH="${REPO_ROOT}/tools/cuda-tile-tblgen/BytecodeGenUtilities.cpp"
OPS_TD_PATH="${REPO_ROOT}/include/cuda_tile/Dialect/CudaTile/IR/Ops.td"
CUDATILE_CPP_PATH="${REPO_ROOT}/lib/Dialect/CudaTile/IR/CudaTile.cpp"

echo "[patch] repo_root=${REPO_ROOT}"

# The released 13.4 dialect references an MLIR float type absent from public
# LLVM (including cuda-tile's own pin). Triton has no frontend type for it.
# Omit only that unsupported type; retain bytecode tags for all supported types.
if ! grep -q 'Float8E5M3FNUType' "${LLVM_SYSPATH}/include/mlir/IR/BuiltinTypes.h.inc"; then
  dtype_patch="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cuda_tile_13_4_public_llvm.patch"
  if git -C "${REPO_ROOT}" apply --reverse --check "${dtype_patch}" 2>/dev/null; then
    echo "[patch] Public LLVM dtype compatibility already applied"
  else
    git -C "${REPO_ROOT}" apply --check "${dtype_patch}"
    git -C "${REPO_ROOT}" apply "${dtype_patch}"
  fi
fi

# 1) Patch BytecodeGenUtilities.cpp for LLVM api changes:
# Replace "getArgToOperandOrAttribute" with "getArgToOperandAttrOrProp"
# and "OperandOrAttribute" with "OperandAttrOrProp".
if [[ -f "${BYTECODE_UTIL_PATH}" ]]; then
  echo "[patch] Patching: ${BYTECODE_UTIL_PATH}"
  patch_in_place "${BYTECODE_UTIL_PATH}" \
    -e 's/getArgToOperandOrAttribute/getArgToOperandAttrOrProp/g' \
    -e 's/OperandOrAttribute/OperandAttrOrProp/g'
fi

# 2) Patch Ops.td for LLVM api changes:
# - replace 'CArg<"ValueRange", "std::nullopt">:$initArgs' with 'CArg<"ValueRange", "{}">:$initArgs'
# - replace 'build($_builder, $_state, std::nullopt)' with 'build($_builder, $_state, ::mlir::ValueRange{})'
if [[ -f "${OPS_TD_PATH}" ]]; then
  echo "[patch] Patching: ${OPS_TD_PATH}"
  patch_in_place "${OPS_TD_PATH}" \
    -e 's/CArg<"ValueRange", "std::nullopt">:$initArgs/CArg<"ValueRange", "{}">:$initArgs/g' \
    -e 's/build($_builder, $_state, std::nullopt)/build($_builder, $_state, ::mlir::ValueRange{})/g'
fi

# Public LLVM represents symbol naming through SymbolOpInterface. The release
# source also names a SymbolName trait not available in this LLVM revision.
for symbol_td in "${OPS_TD_PATH}" "${REPO_ROOT}/include/cuda_tile/Dialect/CudaTile/IR/TestingOps.td"; do
  patch_in_place "${symbol_td}" \
    -e 's/Symbol, SymbolName/Symbol/g' \
    -e 's/\bSymbolName\b/Symbol/g'
done

# 3) Patch CudaTile.cpp for LLVM api changes:
# replace 'ValueRange(), /*attributes=*/std::nullopt)' with
# 'ValueRange(), /*attributes=*/llvm::ArrayRef<mlir::NamedAttribute>{})'
if [[ -f "${CUDATILE_CPP_PATH}" ]]; then
  echo "[patch] Patching: ${CUDATILE_CPP_PATH}"
  patch_in_place "${CUDATILE_CPP_PATH}" \
    -e 's|ValueRange(), /\*attributes=\*/std::nullopt)|ValueRange(), /\*attributes=\*/llvm::ArrayRef<mlir::NamedAttribute>{})|g'
fi

# Operation properties were renamed in newer LLVM revisions.
if grep -q 'class PropertyRef' "${LLVM_SYSPATH}/include/mlir/IR/OperationSupport.h"; then
  patch_in_place "${CUDATILE_CPP_PATH}" -e 's/\bOpaqueProperties\b/PropertyRef/g'
else
  patch_in_place "${CUDATILE_CPP_PATH}" -e 's/\bPropertyRef\b/OpaqueProperties/g'
fi

# 4) New LLVM revisions renamed DenseIntOrFPElementsAttr to
# DenseTypedElementsAttr. Keep TileIR 13.4 pinned and bridge the copied build
# source instead of changing the released TileIR sources.
if grep -q 'DenseTypedElementsAttr' "${LLVM_SYSPATH}/include/mlir/IR/BuiltinAttributes.h"; then
  dense_rename='s/DenseIntOrFPElementsAttr/DenseTypedElementsAttr/g'
else
  dense_rename='s/DenseTypedElementsAttr/DenseIntOrFPElementsAttr/g'
fi
while IFS= read -r -d '' source; do
  patch_in_place "${source}" -e "${dense_rename}"
done < <(find "${REPO_ROOT}" -path "${REPO_ROOT}/build" -prune -o \
  -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -print0)

# CUDA Tile 13.4 provides canonical i1 bytecode encoding and decoding.
# Keep that upstream implementation intact.

# 5) Patch BytecodeReader.cpp for LLVM api changes:
# Select the API exposed by this Triton revision's pinned LLVM headers.
BYTECODE_READER_PATH="${REPO_ROOT}/lib/Bytecode/Reader/BytecodeReader.cpp"
if [[ -f "${BYTECODE_READER_PATH}" ]]; then
  echo "[patch] Patching: ${BYTECODE_READER_PATH}"
  if grep -q 'make_scope_exit' "${LLVM_SYSPATH}/include/llvm/ADT/ScopeExit.h"; then
    patch_in_place "${BYTECODE_READER_PATH}" \
      -e 's/llvm::scope_exit removeIndex(/auto removeIndex = llvm::make_scope_exit(/g'
  else
    patch_in_place "${BYTECODE_READER_PATH}" \
      -e 's/auto removeIndex = llvm::make_scope_exit(/llvm::scope_exit removeIndex(/g'
  fi
  if grep -q 'bool &detectedSplat' "${LLVM_SYSPATH}/include/mlir/IR/BuiltinAttributes.h"; then
    if ! grep -q 'bool isSplat = false;' "${BYTECODE_READER_PATH}"; then
      patch_in_place "${BYTECODE_READER_PATH}" \
        -e '/if (!DenseElementsAttr::isValidRawBuffer(tileType, rawData/i\      [[maybe_unused]] bool isSplat = false;'
    fi
    patch_in_place "${BYTECODE_READER_PATH}" \
      -e 's/DenseElementsAttr::isValidRawBuffer(tileType, rawData)/DenseElementsAttr::isValidRawBuffer(tileType, rawData, isSplat)/g'
  else
    patch_in_place "${BYTECODE_READER_PATH}" \
      -e 's/DenseElementsAttr::isValidRawBuffer(tileType, rawData, isSplat)/DenseElementsAttr::isValidRawBuffer(tileType, rawData)/g'
  fi
fi

# This option is newer than the LLVM pinned by Triton 3.7.
DIALECT_TD_PATH="${REPO_ROOT}/include/cuda_tile/Dialect/CudaTile/IR/Dialect.td"
if [[ -f "${DIALECT_TD_PATH}" ]] && grep -q 'usePropertiesForAttributes' "${DIALECT_TD_PATH}"; then
  echo "[patch] Removing unsupported usePropertiesForAttributes"
  patch_in_place "${DIALECT_TD_PATH}" -e '/usePropertiesForAttributes/d'
fi

echo "[patch] DONE"
