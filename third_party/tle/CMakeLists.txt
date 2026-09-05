
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
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
  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_raw.cc
    LINK_LIBS
    MLIRTargetLLVMIRImport
    TritonTLERawUtils
    TritonTLETransforms
    TritonIR
    )

  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport)
  add_dependencies(TritonTLE TleTableGen TritonTLETransformsIncGen)
endif()

add_subdirectory(test)
