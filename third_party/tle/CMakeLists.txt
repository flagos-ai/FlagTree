
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


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

# FlagPrism: package builds disable Triton C++ unit tests, so do not register
# TLE lit tests that depend on the test-only triton-opt targets in that mode.
if(TRITON_BUILD_UT)
  add_subdirectory(test)
endif()
