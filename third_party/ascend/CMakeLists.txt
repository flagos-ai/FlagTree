include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR. The subdirectory is added from the
# top-level CMakeLists (flagtree_configure_ascend_npu_ir_dependency), which also
# sets ASCENDNPU_IR_SRC_DIR / ASCENDNPU_IR_BINARY_DIR and the BISHENGIR_* options,
# so that a source dir is only ever bound to one binary dir.
if(NOT ASCENDNPU_IR_SRC_DIR)
  message(FATAL_ERROR
    "ASCENDNPU_IR_SRC_DIR is unset; the ascend backend expects the top-level "
    "CMakeLists to have configured AscendNPU-IR already.")
endif()

set(TRITON_DISTRIBUTED_ASCEND_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Triton-distributed-ascend" CACHE PATH
    "Path to Triton-distributed-ascend source root")
set(TRITON_DISTRIBUTED_ASCEND_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/Triton-distributed-ascend")

include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

if(TRITON_BUILD_TD)
  if(NOT EXISTS "${TRITON_DISTRIBUTED_ASCEND_SRC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Triton-Distributed submodule is not initialized: ${TRITON_DISTRIBUTED_ASCEND_SRC_DIR}")
  endif()

  set(TRITON_USE_ASCEND ON)
  add_compile_definitions(TRITON_USE_ASCEND=1)

  if(TRITON_BUILD_PYTHON_MODULE)
    # Python3::Module / pybind11::headers targets are created in the root
    # CMakeLists.txt *after* this directory is processed, so resolve them here
    # when the distributed plugin needs them.
    if(NOT Python3_FOUND OR NOT TARGET Python3::Module)
      find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    endif()
    if(NOT pybind11_FOUND)
      find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
    endif()
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${pybind11_INCLUDE_DIR})
  endif()

  include_directories(${TRITON_DISTRIBUTED_ASCEND_SRC_DIR}/include)
  include_directories(${TRITON_DISTRIBUTED_ASCEND_BINARY_DIR}/include)
  add_subdirectory(${TRITON_DISTRIBUTED_ASCEND_SRC_DIR}/include/TritonDistributed
                   ${TRITON_DISTRIBUTED_ASCEND_BINARY_DIR}/include/TritonDistributed)
  add_subdirectory(${TRITON_DISTRIBUTED_ASCEND_SRC_DIR}/lib
                   ${TRITON_DISTRIBUTED_ASCEND_BINARY_DIR}/lib)

  if(TRITON_BUILD_PYTHON_MODULE)
    set(PYTHON_DIST_SRC_PATH ${TRITON_DISTRIBUTED_ASCEND_SRC_DIR}/python/src)
    add_triton_plugin(TritonDistributed
      ${PYTHON_DIST_SRC_PATH}/triton_distributed.cc
      ${PYTHON_DIST_SRC_PATH}/ascend_passes.cc
      ${PYTHON_DIST_SRC_PATH}/ir.cc)
    target_link_libraries(TritonDistributed PRIVATE DistributedIR Python3::Module pybind11::headers)
  endif()
endif()

option(TRITON_ASCEND_ENABLE_COSTMODEL_INPROC "Enable in-process costmodel C++ bridge" ON)
if(TRITON_ASCEND_ENABLE_COSTMODEL_INPROC)
  set(TRITON_ASCEND_COSTMODEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/costmodel" CACHE PATH "Path to vendored vTriton source root")
  set(_costmodel_include_dir "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel")
  set(_costmodel_lib_dir "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/lib/AscendModel")

  if(EXISTS "${_costmodel_include_dir}/CMakeLists.txt" AND EXISTS "${_costmodel_lib_dir}/CMakeLists.txt")
    # Generate HardwareParams.h required by AscendModel analysis/transforms.
    set(ASCEND_CUBE_TFLOPS 320 CACHE STRING "Cube Core FP16 TFLOPS")
    set(ASCEND_CUBE_M 16 CACHE STRING "Cube Core M dimension")
    set(ASCEND_CUBE_N 16 CACHE STRING "Cube Core N dimension")
    set(ASCEND_CUBE_K 16 CACHE STRING "Cube Core K dimension")
    set(ASCEND_VECTOR_TFLOPS 10 CACHE STRING "Vector Core FP32 TFLOPS")
    set(ASCEND_VECTOR_WIDTH 256 CACHE STRING "Vector width in bits")
    set(ASCEND_HBM_BW 1555 CACHE STRING "HBM Bandwidth GB/s")
    set(ASCEND_L2_SIZE 192 CACHE STRING "L2 Cache Size MB")
    set(ASCEND_L1_SIZE 1024 CACHE STRING "L1 Buffer Size KB")
    set(_costmodel_gen_include "${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include")
    file(MAKE_DIRECTORY "${_costmodel_gen_include}/AscendModel")
    configure_file(
      "${_costmodel_include_dir}/HardwareParams.h.in"
      "${_costmodel_gen_include}/AscendModel/HardwareParams.h"
      @ONLY
    )
    include_directories("${_costmodel_gen_include}")
    # TableGen outputs (*.inc) from AscendModel/include land in this binary dir.
    include_directories("${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel")
    include_directories("${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include")

    add_subdirectory(${_costmodel_include_dir} ${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel)
    add_subdirectory(${_costmodel_lib_dir} ${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/lib_AscendModel)
    set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 1)
  else()
    message(WARNING "Minimal costmodel sources not found under ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}; in-process costmodel bridge disabled")
    set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 0)
  endif()
else()
  set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 0)
endif()

if(TRITON_ASCEND_HAS_INPROC_COSTMODEL)
  target_include_directories(TritonToGraph PRIVATE
    "${_costmodel_gen_include}"
    "${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel"
    "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include"
  )
  target_compile_definitions(TritonToGraph PRIVATE TRITON_ASCEND_HAS_INPROC_COSTMODEL=1)
  target_link_libraries(TritonToGraph PUBLIC AscendModelAnalysis)
else()
  target_compile_definitions(TritonToGraph PRIVATE TRITON_ASCEND_HAS_INPROC_COSTMODEL=0)
endif()

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tle_dsa/tle_ir.cc
  LINK_LIBS
  CommonIRIR
  TritonToGraph
  TritonToLinalg
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRArithDialect
)

target_compile_definitions(TritonAscend PRIVATE __TLE_DSA__)
target_link_libraries(TritonAscend PRIVATE CommonIRToHIVM CommonIRTransforms)
if(TARGET CommonIRIR)
  add_dependencies(TritonAscend CommonIRIR)
endif()

# TLE DSA custom ops (``tle.dsa.ascend.raw`` / ``custom_ops.<arch>.bc``)
#
# The Python registry loads ``custom_ops.<arch-family>.bc`` from the source
# tree. Build every configured family from the device sources when the CANN
# ccec/llvm-link toolchain is available. Otherwise skip the target:
# ``custom_ops/build_custom_ops.sh`` remains the manual path, and non-CANN
# builds keep working.
#
# c220 covers A2/A3, c310 covers A5/Ascend950. A single FlagTree build emits
# both by default so one install can serve either target; the registry picks
# the matching artifact at runtime.
set(ASCEND_CUSTOM_OPS_DIR "${PROJECT_SOURCE_DIR}/python/triton/experimental/tle/language/dsa/ascend/custom_ops")
# Drop artifacts from the old single-architecture build and the removed
# content-addressed aliases; only custom_ops.<family>.bc is used now.
file(GLOB _ASCEND_STALE_CUSTOM_OP_BITCODES
  "${ASCEND_CUSTOM_OPS_DIR}/custom_ops.bc"
  "${ASCEND_CUSTOM_OPS_DIR}/custom_ops.*.*.bc")
if(_ASCEND_STALE_CUSTOM_OP_BITCODES)
  file(REMOVE ${_ASCEND_STALE_CUSTOM_OP_BITCODES})
endif()
set(ASCEND_CUSTOM_OPS_ARCH_FAMILIES "c220;c310" CACHE STRING
  "ccec aicore arch families for TLE custom ops (c220 for A2/A3, c310 for A5)")
if(DEFINED ASCEND_CUSTOM_OPS_ARCH_FAMILY)
  message(WARNING
    "ASCEND_CUSTOM_OPS_ARCH_FAMILY is deprecated and ignored; "
    "use ASCEND_CUSTOM_OPS_ARCH_FAMILIES (default c220;c310)")
endif()
foreach(_family IN LISTS ASCEND_CUSTOM_OPS_ARCH_FAMILIES)
  if(NOT _family MATCHES "^c(220|310)$")
    message(FATAL_ERROR "Unsupported ASCEND_CUSTOM_OPS_ARCH_FAMILIES entry: ${_family}")
  endif()
endforeach()
set(_ASCEND_TEMPLATE_ROOT "${ASCENDNPU_IR_SRC_DIR}/bishengir/lib/Template")
set(_ASCEND_TEMPLATE_INCLUDE "${_ASCEND_TEMPLATE_ROOT}/include")

find_program(ASCEND_CCEC_EXECUTABLE ccec)
if(ASCEND_CCEC_EXECUTABLE)
  get_filename_component(_ASCEND_CCEC_BIN_DIR "${ASCEND_CCEC_EXECUTABLE}" DIRECTORY)
  find_program(ASCEND_LLVM_LINK_EXECUTABLE llvm-link HINTS "${_ASCEND_CCEC_BIN_DIR}")
endif()

if(ASCEND_CCEC_EXECUTABLE AND ASCEND_LLVM_LINK_EXECUTABLE AND EXISTS "${_ASCEND_TEMPLATE_INCLUDE}")
  set(ASCEND_CUSTOM_OPS_BC_DIR "${CMAKE_CURRENT_BINARY_DIR}/ascend_custom_ops")
  set(CCEC_COMMON_FLAGS
    -O2 -x cce --cce-auto-sync=off --cce-aicore-only
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns
    --cce-enable-print --cce-enable-sanitizer -std=c++17)

  # c310 is a reg-based target: its custom-op bodies use the AscendC /
  # reg_compute API, which needs the AscendC headers from the CANN install.
  if("c310" IN_LIST ASCEND_CUSTOM_OPS_ARCH_FAMILIES)
    set(ASCENDC_ROOT "$ENV{ASCEND_HOME_PATH}/asc" CACHE PATH
      "AscendC headers used by the c310 custom-op bodies")
    if(NOT EXISTS "${ASCENDC_ROOT}/impl/basic_api")
      get_filename_component(_ASCEND_CCEC_PREFIX "${ASCEND_CCEC_EXECUTABLE}" DIRECTORY)
      get_filename_component(_ASCEND_CCEC_PREFIX "${_ASCEND_CCEC_PREFIX}" DIRECTORY)
      set(ASCENDC_ROOT "${_ASCEND_CCEC_PREFIX}/asc")
    endif()
    if(NOT EXISTS "${ASCENDC_ROOT}/impl/basic_api")
      message(FATAL_ERROR
        "AscendC headers not found for the c310 custom ops; set -DASCENDC_ROOT=<cann>/asc")
    endif()
  endif()

  set(ASCEND_CUSTOM_OPS_BITCODES "")
  foreach(family IN LISTS ASCEND_CUSTOM_OPS_ARCH_FAMILIES)
    file(MAKE_DIRECTORY "${ASCEND_CUSTOM_OPS_BC_DIR}/${family}")

    set(_family_flags ${CCEC_COMMON_FLAGS})
    if(family STREQUAL "c310")
      list(APPEND _family_flags --cce-mask-opt
        -DTILING_KEY_VAR=0
        -I "${ASCENDC_ROOT}"
        -I "${ASCENDC_ROOT}/impl/basic_api"
        -I "${ASCENDC_ROOT}/include"
        -I "${ASCENDC_ROOT}/include/basic_api")
    endif()

    # Each custom op is implemented in its own .cpp and compiled into its own
    # bitcode: "<src>::<arch>" — arch is the ccec aicore target for that op.
    set(_family_ops
      "mem_ops/duplicate.cpp::dav-${family}-vec"
      "mem_ops/gather_gm_to_l1.cpp::dav-${family}-cube"
      "mem_ops/gather_gm_to_ub.cpp::dav-${family}-vec"
      "mem_ops/gather_mask.cpp::dav-${family}-vec"
      "reduction_ops/pair_reduce_sum.cpp::dav-${family}-vec"
      "sort_ops/sort32.cpp::dav-${family}-vec"
      "sort_ops/sort_1d_pack.cpp::dav-${family}-vec"
      "sort_ops/merge_pack_sort.cpp::dav-${family}-vec"
      "sort_ops/mrgsort.cpp::dav-${family}-vec"
      "sort_ops/unpack_sort.cpp::dav-${family}-vec"
      "mask_ops/compare_scalar.cpp::dav-${family}-vec"
      "cast_ops/cast_int4_to_fp16.cpp::dav-${family}-vec"
    )

    set(_ASCEND_SORT_OPS_DIR "${ASCEND_CUSTOM_OPS_DIR}/sort_ops")
    set(OP_BITCODES "")
    foreach(entry IN LISTS _family_ops)
      string(REPLACE "::" ";" entry_parts "${entry}")
      list(GET entry_parts 0 op_src)
      list(GET entry_parts 1 op_arch)

      # Include order matters: the RegBase headers must win over the mem-based
      # Template headers on c310.
      set(_op_includes -I "${_ASCEND_TEMPLATE_INCLUDE}")
      if(family STREQUAL "c310")
        if(op_arch MATCHES "-cube$")
          list(PREPEND _op_includes -I "${_ASCEND_TEMPLATE_ROOT}/lib/RegBase/Cube/include")
        else()
          list(PREPEND _op_includes -I "${_ASCEND_TEMPLATE_ROOT}/lib/RegBase/Vector/include")
        endif()
      endif()

      set(op_deps "")
      if(op_src MATCHES "^sort_ops/")
        list(APPEND op_deps "${_ASCEND_SORT_OPS_DIR}/sort_common.h")
      endif()
      if(op_src STREQUAL "sort_ops/sort_1d_pack.cpp")
        list(APPEND op_deps
          "${_ASCEND_SORT_OPS_DIR}/sort_base.h"
          "${_ASCEND_SORT_OPS_DIR}/sort_s4096_k129_512.h"
          "${_ASCEND_SORT_OPS_DIR}/sort_s4096_k1_128_k2048.h")
      endif()

      set(op_bc "${ASCEND_CUSTOM_OPS_BC_DIR}/${family}/${op_src}.bc")
      set(op_mix_bc "${ASCEND_CUSTOM_OPS_BC_DIR}/${family}/${op_src}.mix.bc")
      get_filename_component(op_bc_dir "${op_bc}" DIRECTORY)
      file(MAKE_DIRECTORY "${op_bc_dir}")
      list(APPEND OP_BITCODES "${op_bc}" "${op_mix_bc}")
      add_custom_command(
        OUTPUT "${op_bc}"
        COMMAND "${ASCEND_CCEC_EXECUTABLE}" ${_family_flags}
          --cce-aicore-arch=${op_arch}
          ${_op_includes}
          "${ASCEND_CUSTOM_OPS_DIR}/${op_src}"
          -emit-llvm -c -o "${op_bc}"
        DEPENDS "${ASCEND_CUSTOM_OPS_DIR}/${op_src}" ${op_deps}
        COMMENT "ccec ${family} ${op_src} (${op_arch})"
        VERBATIM)
      add_custom_command(
        OUTPUT "${op_mix_bc}"
        COMMAND "${ASCEND_CCEC_EXECUTABLE}" ${_family_flags}
          --cce-aicore-arch=${op_arch}
          -cce-enable-mix -mllvm -enable-mix=true
          ${_op_includes}
          "${ASCEND_CUSTOM_OPS_DIR}/${op_src}"
          -emit-llvm -c -o "${op_mix_bc}"
        DEPENDS "${ASCEND_CUSTOM_OPS_DIR}/${op_src}" ${op_deps}
        COMMENT "ccec ${family} ${op_src} (${op_arch}, mix)"
        VERBATIM)
    endforeach()

    if(family STREQUAL "c310")
      set(_family_templates
        "lib/RegBase/Vector/Sort1D.cpp"
        "lib/RegBase/Vector/Copy1D.cpp")
      set(_template_includes
        -I "${_ASCEND_TEMPLATE_ROOT}/lib/RegBase/Vector/include"
        -I "${_ASCEND_TEMPLATE_INCLUDE}")
    else()
      set(_family_templates
        "lib/Vector/Arange/Arange1D.cpp"
        "lib/Vector/Elementwise/EltWise1D.cpp"
        "lib/Vector/Sort/Sort1D.cpp"
        "lib/DMA/Ubuf/Copy1D.cpp")
      set(_template_includes -I "${_ASCEND_TEMPLATE_INCLUDE}")
    endif()

    set(TEMPLATE_BITCODES "")
    foreach(template_src IN LISTS _family_templates)
      get_filename_component(name_we "${template_src}" NAME_WE)
      set(template_bc "${ASCEND_CUSTOM_OPS_BC_DIR}/${family}/template_${name_we}.bc")
      set(template_mix_bc "${ASCEND_CUSTOM_OPS_BC_DIR}/${family}/template_${name_we}.mix.bc")
      list(APPEND TEMPLATE_BITCODES "${template_bc}" "${template_mix_bc}")
      add_custom_command(
        OUTPUT "${template_bc}"
        COMMAND "${ASCEND_CCEC_EXECUTABLE}" ${_family_flags}
          --cce-aicore-arch=dav-${family}-vec
          ${_template_includes}
          "${_ASCEND_TEMPLATE_ROOT}/${template_src}"
          -emit-llvm -c -o "${template_bc}"
        DEPENDS "${_ASCEND_TEMPLATE_ROOT}/${template_src}"
        COMMENT "ccec ${family} Template ${template_src}"
        VERBATIM)
      add_custom_command(
        OUTPUT "${template_mix_bc}"
        COMMAND "${ASCEND_CCEC_EXECUTABLE}" ${_family_flags}
          --cce-aicore-arch=dav-${family}-vec
          -cce-enable-mix -mllvm -enable-mix=true
          ${_template_includes}
          "${_ASCEND_TEMPLATE_ROOT}/${template_src}"
          -emit-llvm -c -o "${template_mix_bc}"
        DEPENDS "${_ASCEND_TEMPLATE_ROOT}/${template_src}"
        COMMENT "ccec ${family} Template ${template_src} (mix)"
        VERBATIM)
    endforeach()

    set(family_bitcode "${ASCEND_CUSTOM_OPS_DIR}/custom_ops.${family}.bc")
    add_custom_command(
      OUTPUT "${family_bitcode}"
      COMMAND "${ASCEND_LLVM_LINK_EXECUTABLE}" ${OP_BITCODES} ${TEMPLATE_BITCODES}
        -o "${family_bitcode}"
      DEPENDS ${OP_BITCODES} ${TEMPLATE_BITCODES}
      COMMENT "llvm-link -> ${family_bitcode}"
      VERBATIM)
    list(APPEND ASCEND_CUSTOM_OPS_BITCODES "${family_bitcode}")
  endforeach()

  add_custom_target(AscendCustomOpsBitcode DEPENDS ${ASCEND_CUSTOM_OPS_BITCODES})
  add_dependencies(TritonAscend AscendCustomOpsBitcode)
  message(STATUS
    "Ascend custom ops: ${ASCEND_CUSTOM_OPS_ARCH_FAMILIES} bitcode will be built from ${ASCEND_CUSTOM_OPS_DIR}")
else()
  message(STATUS
    "Ascend custom ops: ccec/llvm-link or the AscendNPU-IR Template dir is unavailable; "
    "run python/triton/experimental/tle/language/dsa/ascend/custom_ops/build_custom_ops.sh manually")
endif()

option(TRITON_ENABLE_COVERAGE_LLVM_COV "Enable code llvm-cov coverage tool for Ascend plugin " OFF)
if(TRITON_ENABLE_COVERAGE_LLVM_COV)
  message(STATUS "Enabling llvm-cov coverage tool flags for TritonAscend")
  target_compile_options(TritonAscend PRIVATE
    -fprofile-arcs
    -ftest-coverage
    -O0
    -fprofile-update=atomic
    --coverage
  )
  target_link_options(TritonAscend PRIVATE
    --coverage
    -lgcov
  )
  # branch coverage
  target_compile_definitions(TritonAscend PRIVATE
    COVERAGE_ENABLED=1
  )
endif()


# To enable hitest coverage tool
if(TRITON_ENABLE_COVERAGE_HITEST)
  set_target_properties(TritonAscend PROPERTIES
    RULE_LAUNCH_COMPILE "hitestwrapper"
    RULE_LAUNCH_LINK "hitestwrapper"
  )
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()

if(TRITON_BUILD_TD)
  add_subdirectory(Triton-distributed-ascend/csrc)
endif()

if(TRITON_ASCEND_HAS_INPROC_COSTMODEL)
  target_compile_definitions(TritonAscend PRIVATE TRITON_ASCEND_HAS_INPROC_COSTMODEL=1)
  target_link_libraries(TritonAscend PRIVATE
    AscendModelIR
    AscendModelTransforms
    AscendModelAnalysis
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
    LLVMSupport
  )
else()
  target_compile_definitions(TritonAscend PRIVATE TRITON_ASCEND_HAS_INPROC_COSTMODEL=0)
endif()
