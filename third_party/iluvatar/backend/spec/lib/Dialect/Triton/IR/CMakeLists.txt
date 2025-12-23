add_triton_library(FlagTree_iluvatar_TritonIR
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
