add_triton_library(FlagTree_iluvatar_TritonIR
  Ops.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
