add_triton_library(FlagTree_iluvatar_TritonConversionTritonGPUToLLVM
  ElementwiseOpToLLVM.cpp
  TypeConverter.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
