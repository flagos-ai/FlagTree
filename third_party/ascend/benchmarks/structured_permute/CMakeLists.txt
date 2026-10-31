cmake_minimum_required(VERSION 3.20)
project(structured_permute LANGUAGES C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.0.0" CACHE PATH "CANN toolkit")
set(SOC_VERSION "Ascend910B4" CACHE STRING "Target SoC")
set(RUN_MODE "npu")
include("${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
ascendc_library(permute_device STATIC aiv.cpp)
add_library(permute_reference SHARED launch.cpp)
target_link_libraries(permute_reference PRIVATE permute_device)

# Build each independent reference as its own SDK device object.
ascendc_library(matrix_transport_device STATIC matrix_transport_aiv.cpp)
add_library(matrix_transport_reference SHARED matrix_transport_launch.cpp)
target_link_libraries(matrix_transport_reference PRIVATE matrix_transport_device)

# Opt-in ABI contract tests link the same packaged device bitcode as Triton.
set(FLAGTREE_DEVICE_BITCODE "" CACHE FILEPATH "Packaged custom_ops.bc for UB ABI tests")
if(FLAGTREE_DEVICE_BITCODE)
  if(NOT EXISTS "${FLAGTREE_DEVICE_BITCODE}")
    message(FATAL_ERROR "Missing FLAGTREE_DEVICE_BITCODE: ${FLAGTREE_DEVICE_BITCODE}")
  endif()
  # Put dependency digests into the primary source consumed by the SDK's
  # external build. Replacing bitcode must rebuild the test device binary.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${FLAGTREE_DEVICE_BITCODE}" "${CMAKE_CURRENT_SOURCE_DIR}/ub_contract.cpp")
  file(SHA256 "${FLAGTREE_DEVICE_BITCODE}" _ub_bc_hash)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/ub_contract.cpp" _ub_src_hash)
  file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ub_contract_versioned.cpp"
    CONTENT "// bitcode: ${_ub_bc_hash}\n// source: ${_ub_src_hash}\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/ub_contract.cpp\"\n" @ONLY)
  ascendc_library(ub_contract_device STATIC "${CMAKE_CURRENT_BINARY_DIR}/ub_contract_versioned.cpp")
  ascendc_compile_options(ub_contract_device PRIVATE
    "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${FLAGTREE_DEVICE_BITCODE}"
    --cce-auto-sync=off)
  add_library(ub_contract_reference SHARED ub_contract_launch.cpp)
  target_link_libraries(ub_contract_reference PRIVATE ub_contract_device)
endif()
