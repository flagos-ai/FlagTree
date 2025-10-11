add_subdirectory(Triton)
add_subdirectory(TritonGPU)

if(NOT FLAGTREE_BACKEND STREQUAL "iluvatar")
  add_subdirectory(TritonNvidiaGPU)
endif()
