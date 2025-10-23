add_triton_library(FlagTree_iluvatar_TritonDialectTritonGPUTransforms
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
