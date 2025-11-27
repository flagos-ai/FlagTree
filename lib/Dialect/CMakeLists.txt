add_subdirectory(Triton)
add_subdirectory(TritonGPU)

if(IS_COMPILE_TritonNvidiaGPU)
  add_subdirectory(TritonNvidiaGPU)
endif()
