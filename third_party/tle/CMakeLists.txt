# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
add_subdirectory(dialect)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLE ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc LINK_LIBS TritonTLETransforms TritonIR)
  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers)
endif()

add_subdirectory(test)
