#!/usr/bin/env bash
# Copyright 2026- Xcoresigma Technology Co., Ltd
# Manual rebuild of custom_ops.<arch>.bc. The CMake build also drives ccec/llvm-link
# (third_party/ascend/CMakeLists.txt) when the CANN toolchain is available;
# this script is the fallback for machines where it is not found at configure
# time, or for rebuilding after editing a device source.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# ``custom_ops.<arch>.bc`` must be built for the same ccec arch family as the kernel:
# c220 for A2/A3, c310 for A5/Ascend950. Export CUSTOM_OPS_ARCH_FAMILY=c310
# when building for a 950 target.
ARCH_FAMILY="${CUSTOM_OPS_ARCH_FAMILY:-c220}"
case "${ARCH_FAMILY}" in
  c220|c310) ;;
  *)
    echo "error: unsupported CUSTOM_OPS_ARCH_FAMILY='${ARCH_FAMILY}' (expected c220 or c310)" >&2
    exit 1
    ;;
esac
CUSTOM_OPS_BC="${SCRIPT_DIR}/custom_ops.${ARCH_FAMILY}.bc"
# Drop artifacts from the old single-architecture build and the removed
# content-addressed aliases; only custom_ops.<family>.bc is used now.
rm -f "${SCRIPT_DIR}/custom_ops.bc" "${SCRIPT_DIR}"/custom_ops.*.*.bc
TEMPLATE_ROOT="${SCRIPT_DIR}/../../../../../../../../third_party/ascend/AscendNPU-IR/bishengir/lib/Template"
TEMPLATE_INCLUDE="${TEMPLATE_ROOT}/include"
TEMPLATE_LIB="${TEMPLATE_ROOT}/lib"
TEMPLATE_BC=(
  "${SCRIPT_DIR}/template_arange.bc"
  "${SCRIPT_DIR}/template_eltwise1d.bc"
  "${SCRIPT_DIR}/template_sort1d.bc"
  "${SCRIPT_DIR}/template_copy1d.bc"
)

# Keep CCEC/LLVM_LINK from the same CANN/BiShengIR release as the
# ``bishengir-compile`` used by Triton; mixing releases can produce
# "Cannot select" intrinsic or data-layout errors when linking device bitcode.
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

# c310 is a reg-based target where the legacy mem-based intrinsics are not
# available. The c310 custom-op bodies use the AscendC/reg_compute API, so
# locate the AscendC headers from the same CANN installation as ccec.
ASCENDC_ROOT="${ASCENDC_ROOT:-${ASCEND_HOME_PATH:-}/asc}"
if [[ ! -d "${ASCENDC_ROOT}/impl/basic_api" ]]; then
  ASCENDC_ROOT="$(dirname -- "$(dirname -- "${CCEC}")")/asc"
fi
ASCENDC_EXTRA_ARGS=()
if [[ "${ARCH_FAMILY}" == "c310" ]]; then
  if [[ ! -d "${ASCENDC_ROOT}/impl/basic_api" ]]; then
    echo "error: AscendC headers not found under '${ASCENDC_ROOT}'; set ASCENDC_ROOT" >&2
    exit 1
  fi
  ASCENDC_EXTRA_ARGS=(
    --cce-mask-opt
    # Avoid the tentative `g_tilingKey` global in every translation unit
    # (asc-devkit's own device build defines TILING_KEY_VAR the same way).
    -DTILING_KEY_VAR=0
    -I "${ASCENDC_ROOT}"
    -I "${ASCENDC_ROOT}/impl/basic_api"
    -I "${ASCENDC_ROOT}/include"
    -I "${ASCENDC_ROOT}/include/basic_api"
  )
fi

# c310 builds the shared Template helpers from the RegBase tree; the mem-based
# lib/Vector sources are c220-only.
TEMPLATE_INCLUDES=( -I "${TEMPLATE_INCLUDE}" )
if [[ "${ARCH_FAMILY}" == "c310" ]]; then
  TEMPLATE_SOURCES=(
    "${TEMPLATE_ROOT}/lib/RegBase/Vector/Sort1D.cpp"
    "${TEMPLATE_ROOT}/lib/RegBase/Vector/Copy1D.cpp"
  )
else
  TEMPLATE_SOURCES=(
    "${TEMPLATE_LIB}/Vector/Arange/Arange1D.cpp"
    "${TEMPLATE_LIB}/Vector/Elementwise/EltWise1D.cpp"
    "${TEMPLATE_LIB}/Vector/Sort/Sort1D.cpp"
    "${TEMPLATE_LIB}/DMA/Ubuf/Copy1D.cpp"
  )
fi

# Each custom op is implemented in its own .cpp and compiled into its own
# bitcode: "<src>::<arch>" — arch is the ccec aicore target for that op.
CUSTOM_OPS=(
  "mem_ops/duplicate.cpp:dav-${ARCH_FAMILY}-vec"
  "mem_ops/gather_gm_to_l1.cpp:dav-${ARCH_FAMILY}-cube"
  "mem_ops/gather_gm_to_ub.cpp:dav-${ARCH_FAMILY}-vec"
  "mem_ops/gather_mask.cpp:dav-${ARCH_FAMILY}-vec"
  "reduction_ops/pair_reduce_sum.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/sort32.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/sort_1d_pack.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/merge_pack_sort.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/mrgsort.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/unpack_sort.cpp:dav-${ARCH_FAMILY}-vec"
  "mask_ops/compare_scalar.cpp:dav-${ARCH_FAMILY}-vec"
  "cast_ops/cast_int4_to_fp16.cpp:dav-${ARCH_FAMILY}-vec"
)

if [[ ! -d "${TEMPLATE_INCLUDE}" ]]; then
  echo "error: Ascend Template include directory not found: ${TEMPLATE_INCLUDE}" >&2
  exit 1
fi

OP_BC=()
FAILED_OPS=()
# During the c310 port, set ALLOW_PARTIAL=1 to build the bitcode for the ops
# that already have an A5 implementation and skip the rest with a warning.
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
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
    --cce-enable-sanitizer -std=c++17 )
  if [[ "${ARCH_FAMILY}" == "c310" ]]; then
    case "${arch}" in
      *-cube) REGBASE_INCLUDE="${TEMPLATE_ROOT}/lib/RegBase/Cube/include" ;;
      *)      REGBASE_INCLUDE="${TEMPLATE_ROOT}/lib/RegBase/Vector/include" ;;
    esac
    CCEC_COMMON_ARGS+=( -I "${REGBASE_INCLUDE}" )
  fi
  CCEC_COMMON_ARGS+=( "${TEMPLATE_INCLUDES[@]}" "${ASCENDC_EXTRA_ARGS[@]}" )

  bc="${SCRIPT_DIR}/${src%.cpp}.bc"
  mix_bc="${SCRIPT_DIR}/${src%.cpp}.mix.bc"

  echo "== ccec ${src} (${arch})"
  if ! "${CCEC}" "${CCEC_COMMON_ARGS[@]}" \
    "${SCRIPT_DIR}/${src}" -emit-llvm -c -o "${bc}"; then
    if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
      echo "warning: skipping ${src} (${arch}); no c310 implementation yet" >&2
      FAILED_OPS+=("${src}")
      rm -f "${bc}" "${mix_bc}"
      continue
    fi
    exit 1
  fi

  echo "== ccec ${src} (${arch}, mix)"
  if ! "${CCEC}" "${CCEC_COMMON_ARGS[@]}" -cce-enable-mix -mllvm -enable-mix=true \
    "${SCRIPT_DIR}/${src}" -emit-llvm -c -o "${mix_bc}"; then
    if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
      echo "warning: skipping ${src} (${arch}, mix)" >&2
      FAILED_OPS+=("${src} (mix)")
      rm -f "${bc}" "${mix_bc}"
      continue
    fi
    exit 1
  fi
  OP_BC+=("${bc}" "${mix_bc}")
done

if [[ "${#FAILED_OPS[@]}" -gt 0 ]]; then
  echo "warning: custom_ops.${ARCH_FAMILY}.bc omits ops without a ${ARCH_FAMILY} implementation:" >&2
  printf '  - %s\n' "${FAILED_OPS[@]}" >&2
fi

TEMPLATE_BC=()
TEMPLATE_MIX_BC=()
TEMPLATE_CMD_INCLUDES=( "${TEMPLATE_INCLUDES[@]}" )
if [[ "${ARCH_FAMILY}" == "c310" ]]; then
  TEMPLATE_CMD_INCLUDES=( -I "${TEMPLATE_ROOT}/lib/RegBase/Vector/include" "${TEMPLATE_CMD_INCLUDES[@]}" )
fi
for i in "${!TEMPLATE_SOURCES[@]}"; do
  src="${TEMPLATE_SOURCES[$i]}"
  bc="${SCRIPT_DIR}/template_$(basename -- "${src}" .cpp).bc"
  mix_bc="${bc%.bc}.mix.bc"
  TEMPLATE_BC+=("${bc}")
  TEMPLATE_MIX_BC+=("${mix_bc}")
  echo "== ccec ${src}"
  "${CCEC}" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns \
    --cce-aicore-arch=dav-${ARCH_FAMILY}-vec --cce-enable-print \
    --cce-enable-sanitizer -std=c++17 "${TEMPLATE_CMD_INCLUDES[@]}" \
    "${ASCENDC_EXTRA_ARGS[@]}" \
    "${src}" -emit-llvm -c -o "${bc}"
  echo "== ccec ${src} (mix)"
  "${CCEC}" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns \
    --cce-aicore-arch=dav-${ARCH_FAMILY}-vec --cce-enable-print \
    --cce-enable-sanitizer -std=c++17 "${TEMPLATE_CMD_INCLUDES[@]}" \
    "${ASCENDC_EXTRA_ARGS[@]}" \
    -cce-enable-mix -mllvm -enable-mix=true \
    "${src}" -emit-llvm -c -o "${mix_bc}"
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
