
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Slim path: compile only triton_tle_commonir.cc and link flir CommonIRIR.
# Skip the full TLE dialect stack, which mthreads already adds under the same
# binary directories. CommonIR defaults to ON, so an unset FLAGTREE_COMMON_IR
# must take this path. Key off FLAGTREE_COMMON_IR_ENABLED, not a non-empty
# environment variable; only an explicit 0/OFF opts out.
if((("$ENV{FLAGTREE_BACKEND}" STREQUAL "metax") OR ("$ENV{FLAGTREE_BACKEND}" STREQUAL "mthreads")) AND FLAGTREE_COMMON_IR_ENABLED AND TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "[tle] ${FLAGTREE_BACKEND} CommonIR: building CommonIR-only tle binding")
  set(_slim_common_ir_libs "")
  if(TARGET CommonIRIR)
    list(APPEND _slim_common_ir_libs CommonIRIR)
  endif()
  # [metax+CommonIR] 编译并链接通用 CommonIRToTTGIR(仅 Arith/CommonIRIR/TritonIR/TritonGPUIR,无 Hopper)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
  if(TARGET CommonIRIR)
    add_subdirectory(include/CommonIRToTTGIR)
    add_subdirectory(lib/CommonIRToTTGIR)
  endif()
  if(TARGET TleCommonIRToTTGIR)
    list(APPEND _slim_common_ir_libs TleCommonIRToTTGIR)
  endif()
  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_commonir.cc
    LINK_LIBS TritonIR ${_slim_common_ir_libs})
  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers)
  if(TARGET CommonIRTableGen)
    add_dependencies(TritonTLE CommonIRTableGen CommonIRAttrDefsIncGen CommonIRTypesIncGen)
  endif()
  if(TARGET TleCommonIRToTTGIRPassIncGen)
    add_dependencies(TritonTLE TleCommonIRToTTGIRPassIncGen)
  endif()
  # bin/ 的 CLI 工具经 TRITON_LIBS 仍引用这三个老 tle 库名（瘦身路不建它们）。
  # 建三个空 stub 库满足链接（bin 已剥 __TLE__，不需要它们的符号）。
  set(_tle_stub "${CMAKE_CURRENT_BINARY_DIR}/tle_stub.cpp")
  file(WRITE "${_tle_stub}" "")
  foreach(_stub TritonTLEAnalysis TleToLLVM TritonTLETransforms)
    if(NOT TARGET ${_stub})
      add_library(${_stub} STATIC "${_tle_stub}")
    endif()
  endforeach()
  return()
endif()


# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

if(TARGET TritonGPUCTAAttrIncGen)
  list(APPEND TLE_TRITON_GPU_ATTR_DEPENDS TritonGPUCTAAttrIncGen)
elseif(TARGET TritonGPUCGAAttrIncGen)  # mthreads
  list(APPEND TLE_TRITON_GPU_ATTR_DEPENDS TritonGPUCGAAttrIncGen)
endif()
if(TARGET TritonGPUOpsEnumsIncGen)  # mthreads
  list(APPEND TLE_TRITON_GPU_ATTR_DEPENDS TritonGPUOpsEnumsIncGen)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flagcx/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flagcx/include)
endif()

# eco-tle dsa sub-project
set(TLE_INCLUDE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(TLE_INCLUDE_BINARY "${CMAKE_CURRENT_BINARY_DIR}/include")
include_directories(${TLE_INCLUDE_SOURCE})
include_directories(${TLE_INCLUDE_BINARY})
add_subdirectory(include/tle-dsa/Dialect/IR)
add_subdirectory(lib/Dialect/IR)
add_subdirectory(lib/Conversion/DsaToCore)
add_subdirectory(python)

# Suppress C++20 lambda-capture extension warning when building against newer LLVM.
add_compile_options(-Wno-c++20-extensions)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flagcx/adaptor/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flagcx/adaptor/include)
endif()
add_subdirectory(dialect)
add_subdirectory(utils)
if(TRITON_BUILD_PYTHON_MODULE)
  set(TLE_COMMON_IR_LINK_LIBS "")
  if(FLAGTREE_COMMON_IR_ENABLED)
    if(TARGET CommonIRIR)
      list(APPEND TLE_COMMON_IR_LINK_LIBS CommonIRIR)
    endif()
    if(TARGET CommonIRTransforms)
      list(APPEND TLE_COMMON_IR_LINK_LIBS CommonIRTransforms)
    endif()
  endif()

  if(FLAGTREE_COMMON_IR_ENABLED AND TARGET CommonIRIR)
    add_subdirectory(include/CommonIRToTTGIR)
    add_subdirectory(lib/CommonIRToTTGIR)
  endif()

  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_raw.cc
    LINK_LIBS
    MLIRTargetLLVMIRImport
    TritonTLERawUtils
    TritonTLETransforms
    TritonIR
    ${TLE_COMMON_IR_LINK_LIBS}
    )

  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport)
  if(FLAGTREE_COMMON_IR_ENABLED AND TARGET TleCommonIRToTTGIR)
    target_link_libraries(TritonTLE PRIVATE TleCommonIRToTTGIR)
  endif()
  add_dependencies(TritonTLE
    TleTableGen
    TritonTLETransformsIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    ${TLE_TRITON_GPU_ATTR_DEPENDS}
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
  )
  if(FLAGTREE_COMMON_IR_ENABLED)
    add_dependencies(TritonTLE
      CommonIRTableGen
      CommonIRAttrDefsIncGen
      CommonIRTypesIncGen
    )
  endif()
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(test)
endif()
