add_triton_library(FlagTree_iluvatar_TritonConversionTritonGPUToLLVM
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)