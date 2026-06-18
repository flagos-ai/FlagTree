
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
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
