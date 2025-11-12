add_triton_library(FlagTree_iluvatar_TritonLLVMIR
  LLVMDIScope.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
