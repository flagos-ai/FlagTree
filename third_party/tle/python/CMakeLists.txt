if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTleDsaTemplate
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_dsa.cc
    LINK_LIBS TleDsaIR
  )
  target_link_libraries(TritonTleDsaTemplate PRIVATE Python3::Module pybind11::headers)
endif()
