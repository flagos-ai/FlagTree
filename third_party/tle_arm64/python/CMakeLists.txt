if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTleArm64Template
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_arm64.cc
    LINK_LIBS TritonIR
    DEPENDS TritonIR
  )
  target_link_libraries(TritonTleArm64Template PRIVATE Python3::Module pybind11::headers)
endif()
