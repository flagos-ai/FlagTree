include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)
add_subdirectory(scripts)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonThrive
      ${CMAKE_CURRENT_SOURCE_DIR}/plugin/triton_thrive.cc
  )
  target_link_libraries(TritonThrive PRIVATE Python3::Module pybind11::headers)
endif()
