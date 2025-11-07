add_triton_library(FlagTree_iluvatar_TritonDialectTritonIR
  Ops.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
