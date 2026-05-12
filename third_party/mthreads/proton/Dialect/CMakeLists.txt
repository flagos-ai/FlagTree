include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  set(PROTON_PLUGIN_LINK_LIBS
    ProtonToProtonGPU
    ProtonGPUToLLVM
    ProtonNVIDIAGPUToLLVM
    ProtonAnalysis
  )
  if(TRITON_ENABLE_AMD)
    list(APPEND PROTON_PLUGIN_LINK_LIBS ProtonAMDGPUToLLVM)
  endif()
  add_triton_plugin(TritonProton ${CMAKE_CURRENT_SOURCE_DIR}/triton_proton.cc LINK_LIBS ${PROTON_PLUGIN_LINK_LIBS})
  target_link_libraries(TritonProton PRIVATE Python3::Module pybind11::headers)
endif()
