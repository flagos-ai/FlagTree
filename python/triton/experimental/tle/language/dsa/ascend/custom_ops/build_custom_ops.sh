#!/usr/bin/env bash
# Copyright 2026- Xcoresigma Technology Co., Ltd
# Manual rebuild of custom_ops.bc — the normal build drives ccec/llvm-link
# directly from third_party/tle/dsa/dialect/lib/CMakeLists.txt instead.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_OPS_BC="${SCRIPT_DIR}/custom_ops.bc"
TEMPLATE_ROOT="${SCRIPT_DIR}/../../../../../../../../third_party/ascend/AscendNPU-IR/bishengir/lib/Template"
TEMPLATE_INCLUDE="${TEMPLATE_ROOT}/include"
TEMPLATE_LIB="${TEMPLATE_ROOT}/lib"
TEMPLATE_BC=(
  "${SCRIPT_DIR}/template_arange.bc"
  "${SCRIPT_DIR}/template_eltwise1d.bc"
  "${SCRIPT_DIR}/template_sort1d.bc"
  "${SCRIPT_DIR}/template_copy1d.bc"
)

CCEC="${CCEC:-$(command -v ccec || true)}"
if [[ -z "${CCEC}" ]]; then
  echo "error: ccec not found; source the CANN environment first" >&2
  exit 1
fi
CCEC="$(readlink -f "${CCEC}")"

CCEC_BIN_DIR="$(dirname -- "${CCEC}")"
LLVM_LINK="${LLVM_LINK:-${CCEC_BIN_DIR}/llvm-link}"
if [[ ! -x "${LLVM_LINK}" ]]; then
  echo "error: CANN llvm-link not found: ${LLVM_LINK}" >&2
  exit 1
fi

# Each custom op is implemented in its own .cpp and compiled into its own
# bitcode: "<src>::<arch>" — arch is the ccec aicore target for that op.
CUSTOM_OPS=(
  "mem_ops/gather_gm_to_l1.cpp:dav-c220-cube"
  "mem_ops/gather_gm_to_ub.cpp:dav-c220-vec"
  "sort_ops/sort_1d_pack.cpp:dav-c220-vec"
  "sort_ops/merge_pack_sort.cpp:dav-c220-vec"
  "sort_ops/unpack_sort.cpp:dav-c220-vec"
)

if [[ ! -d "${TEMPLATE_INCLUDE}" ]]; then
  echo "error: Ascend Template include directory not found: ${TEMPLATE_INCLUDE}" >&2
  exit 1
fi

OP_BC=()
for entry in "${CUSTOM_OPS[@]}"; do
  src="${entry%%:*}"
  arch="${entry##*:}"
  if [[ ! -f "${SCRIPT_DIR}/${src}" ]]; then
    echo "error: custom-op source not found: ${SCRIPT_DIR}/${src}" >&2
    exit 1
  fi

  CCEC_COMMON_ARGS=( -O2 -x cce --cce-auto-sync=off --cce-aicore-only
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns
    --cce-aicore-arch="${arch}" --cce-enable-print
    --cce-enable-sanitizer -std=c++17 -I "${TEMPLATE_INCLUDE}" )

  bc="${SCRIPT_DIR}/${src%.cpp}.bc"
  mix_bc="${SCRIPT_DIR}/${src%.cpp}.mix.bc"
  OP_BC+=("${bc}" "${mix_bc}")

  echo "== ccec ${src} (${arch})"
  "${CCEC}" "${CCEC_COMMON_ARGS[@]}" \
    "${SCRIPT_DIR}/${src}" -emit-llvm -c -o "${bc}"

  echo "== ccec ${src} (${arch}, mix)"
  "${CCEC}" "${CCEC_COMMON_ARGS[@]}" -cce-enable-mix -mllvm -enable-mix=true \
    "${SCRIPT_DIR}/${src}" -emit-llvm -c -o "${mix_bc}"
done

TEMPLATE_SOURCES=(
  "${TEMPLATE_LIB}/Vector/Arange/Arange1D.cpp"
  "${TEMPLATE_LIB}/Vector/Elementwise/EltWise1D.cpp"
  "${TEMPLATE_LIB}/Vector/Sort/Sort1D.cpp"
  "${TEMPLATE_LIB}/DMA/Ubuf/Copy1D.cpp"
)
TEMPLATE_MIX_BC=()
for i in "${!TEMPLATE_SOURCES[@]}"; do
  echo "== ccec ${TEMPLATE_SOURCES[$i]}"
  "${CCEC}" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns \
    --cce-aicore-arch=dav-c220-vec --cce-enable-print \
    --cce-enable-sanitizer -std=c++17 -I "${TEMPLATE_INCLUDE}" \
    "${TEMPLATE_SOURCES[$i]}" -emit-llvm -c -o "${TEMPLATE_BC[$i]}"
  mix_bc="${TEMPLATE_BC[$i]%.bc}.mix.bc"
  echo "== ccec ${TEMPLATE_SOURCES[$i]} (mix)"
  "${CCEC}" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns \
    --cce-aicore-arch=dav-c220-vec --cce-enable-print \
    --cce-enable-sanitizer -std=c++17 -I "${TEMPLATE_INCLUDE}" \
    -cce-enable-mix -mllvm -enable-mix=true \
    "${TEMPLATE_SOURCES[$i]}" -emit-llvm -c -o "${mix_bc}"
  TEMPLATE_MIX_BC+=("${mix_bc}")
done

echo "== llvm-link -> ${CUSTOM_OPS_BC}"
"${LLVM_LINK}" "${OP_BC[@]}" "${TEMPLATE_BC[@]}" "${TEMPLATE_MIX_BC[@]}" \
  -o "${CUSTOM_OPS_BC}"

if [[ ! -s "${CUSTOM_OPS_BC}" ]]; then
  echo "error: bitcode was not generated: ${CUSTOM_OPS_BC}" >&2
  exit 1
fi

rm -f "${OP_BC[@]}" "${TEMPLATE_BC[@]}" "${TEMPLATE_MIX_BC[@]}"
stat --format='%n %s bytes' "${CUSTOM_OPS_BC}"
sha256sum "${CUSTOM_OPS_BC}"
