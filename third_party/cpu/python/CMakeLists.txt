if(TRITON_BUILD_PYTHON_MODULE)
  # host-CPU LLVM helper bindings. Compiled and linked into libtriton.so as
  # an additional translation unit; the init function is dispatched from
  # main.cc via the standard FlagTree plugin macro.
  add_triton_plugin(TritonHostCpuLlvm
    ${CMAKE_CURRENT_SOURCE_DIR}/host_cpu_llvm.cc
    LINK_LIBS LLVMSupport LLVMTargetParser LLVMMC LLVMIRReader
  )
  target_link_libraries(TritonHostCpuLlvm PRIVATE Python3::Module pybind11::headers)
endif()
