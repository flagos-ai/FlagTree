# Copyright 2026- Xcoresigma Technology Co., Ltd

set(ASCEND_CUSTOM_OPS_DIR "${PROJECT_SOURCE_DIR}/python/triton/experimental/tle/language/dsa/ascend/custom_ops")
set(ASCEND_CUSTOM_OPS_BITCODE "${ASCEND_CUSTOM_OPS_DIR}/custom_ops.bc")

# ccec / llvm-link come from the CANN environment. If they are not on PATH at
# configure time, fall back to bare names resolved from PATH at build time
# (same behaviour as build_custom_ops.sh, kept for manual rebuilds).
find_program(CCEC_EXECUTABLE ccec)
if(NOT CCEC_EXECUTABLE)
  message(STATUS "ccec not found at configure time, deferring lookup to build time")
  set(CCEC_EXECUTABLE ccec)
endif()
get_filename_component(CCEC_BIN_DIR "${CCEC_EXECUTABLE}" DIRECTORY)
find_program(LLVM_LINK_EXECUTABLE llvm-link HINTS "${CCEC_BIN_DIR}")
if(NOT LLVM_LINK_EXECUTABLE)
  set(LLVM_LINK_EXECUTABLE llvm-link)
endif()

set(TEMPLATE_ROOT "${PROJECT_SOURCE_DIR}/third_party/ascend/AscendNPU-IR/bishengir/lib/Template")
set(TEMPLATE_INCLUDE "${TEMPLATE_ROOT}/include")

set(CCEC_COMMON_FLAGS
  -O2 -x cce --cce-auto-sync=off --cce-aicore-only
  --cce-generic-addrspace=off -mllvm -disable-llvm-optzns
  --cce-enable-print --cce-enable-sanitizer -std=c++17)

set(ASCEND_CUSTOM_OPS_BC_DIR "${CMAKE_CURRENT_BINARY_DIR}/ascend_custom_ops")

# Each custom op is implemented in its own .cpp and compiled into its own
# bitcode: "<src>::<arch>" — arch is the ccec aicore target for that op.
set(ASCEND_CUSTOM_OPS
  "mem_ops/gather_gm_to_l1.cpp::dav-c220-cube"
  "mem_ops/gather_gm_to_ub.cpp::dav-c220-vec"
  "sort_ops/sort_1d_pack.cpp::dav-c220-vec"
  "sort_ops/merge_pack_sort.cpp::dav-c220-vec"
  "sort_ops/unpack_sort.cpp::dav-c220-vec"
)

set(SORT_OPS_DIR "${ASCEND_CUSTOM_OPS_DIR}/sort_ops")
set(OP_BITCODES "")
foreach(entry IN LISTS ASCEND_CUSTOM_OPS)
  string(REPLACE "::" ";" entry_parts "${entry}")
  list(GET entry_parts 0 src)
  list(GET entry_parts 1 arch)

  set(op_deps "")
  if(src MATCHES "^sort_ops/")
    list(APPEND op_deps "${SORT_OPS_DIR}/sort_common.h")
  endif()
  if(src STREQUAL "sort_ops/sort_1d_pack.cpp")
    list(APPEND op_deps
      "${SORT_OPS_DIR}/sort_base.h"
      "${SORT_OPS_DIR}/sort_s4096_k129_512.h"
      "${SORT_OPS_DIR}/sort_s4096_k1_128_k2048.h")
  endif()

  set(bc "${ASCEND_CUSTOM_OPS_BC_DIR}/${src}.bc")
  set(mix_bc "${ASCEND_CUSTOM_OPS_BC_DIR}/${src}.mix.bc")
  list(APPEND OP_BITCODES "${bc}" "${mix_bc}")
  add_custom_command(
    OUTPUT "${bc}"
    COMMAND "${CCEC_EXECUTABLE}" ${CCEC_COMMON_FLAGS}
      --cce-aicore-arch=${arch}
      -I "${TEMPLATE_INCLUDE}"
      "${ASCEND_CUSTOM_OPS_DIR}/${src}"
      -emit-llvm -c -o "${bc}"
    DEPENDS "${ASCEND_CUSTOM_OPS_DIR}/${src}" ${op_deps}
    COMMENT "ccec ${src} (${arch})"
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${mix_bc}"
    COMMAND "${CCEC_EXECUTABLE}" ${CCEC_COMMON_FLAGS}
      --cce-aicore-arch=${arch}
      -cce-enable-mix -mllvm -enable-mix=true
      -I "${TEMPLATE_INCLUDE}"
      "${ASCEND_CUSTOM_OPS_DIR}/${src}"
      -emit-llvm -c -o "${mix_bc}"
    DEPENDS "${ASCEND_CUSTOM_OPS_DIR}/${src}" ${op_deps}
    COMMENT "ccec ${src} (${arch}, mix)"
    VERBATIM
  )
endforeach()

set(TEMPLATE_OPS
  "lib/Vector/Arange/Arange1D.cpp"
  "lib/Vector/Elementwise/EltWise1D.cpp"
  "lib/Vector/Sort/Sort1D.cpp"
  "lib/DMA/Ubuf/Copy1D.cpp"
)

set(TEMPLATE_BITCODES "")
foreach(src IN LISTS TEMPLATE_OPS)
  get_filename_component(name_we "${src}" NAME_WE)
  set(bc "${ASCEND_CUSTOM_OPS_BC_DIR}/template_${name_we}.bc")
  set(mix_bc "${ASCEND_CUSTOM_OPS_BC_DIR}/template_${name_we}.mix.bc")
  list(APPEND TEMPLATE_BITCODES "${bc}" "${mix_bc}")
  add_custom_command(
    OUTPUT "${bc}"
    COMMAND "${CCEC_EXECUTABLE}" ${CCEC_COMMON_FLAGS}
      --cce-aicore-arch=dav-c220-vec
      -I "${TEMPLATE_INCLUDE}"
      "${TEMPLATE_ROOT}/${src}"
      -emit-llvm -c -o "${bc}"
    DEPENDS "${TEMPLATE_ROOT}/${src}"
    COMMENT "ccec Template ${src}"
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${mix_bc}"
    COMMAND "${CCEC_EXECUTABLE}" ${CCEC_COMMON_FLAGS}
      --cce-aicore-arch=dav-c220-vec
      -cce-enable-mix -mllvm -enable-mix=true
      -I "${TEMPLATE_INCLUDE}"
      "${TEMPLATE_ROOT}/${src}"
      -emit-llvm -c -o "${mix_bc}"
    DEPENDS "${TEMPLATE_ROOT}/${src}"
    COMMENT "ccec Template ${src} (mix)"
    VERBATIM
  )
endforeach()

add_custom_command(
  OUTPUT "${ASCEND_CUSTOM_OPS_BITCODE}"
  COMMAND "${LLVM_LINK_EXECUTABLE}" ${OP_BITCODES} ${TEMPLATE_BITCODES}
    -o "${ASCEND_CUSTOM_OPS_BITCODE}"
  DEPENDS ${OP_BITCODES} ${TEMPLATE_BITCODES}
  COMMENT "llvm-link -> ${ASCEND_CUSTOM_OPS_BITCODE}"
  VERBATIM
)

add_custom_target(AscendCustomOpsBitcode
  DEPENDS "${ASCEND_CUSTOM_OPS_BITCODE}"
)

add_subdirectory(Analysis)
add_subdirectory(IR)
# Conversion depends on the main TleIR target, so it is added by
# third_party/tle/CMakeLists.txt after dialect/lib is configured.
# add_subdirectory(Conversion)
