if (FLAGTREE_BACKEND)
    set(NVGPUIR "")
else()
    set(NVGPUIR "TritonNvidiaGPUIR")
endif()

get_flagtree_backend_lib("TritonAnalysis" _EXTRA_LINK_LIBS)

add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
