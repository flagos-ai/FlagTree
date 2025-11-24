add_triton_library(FlagTree_iluvatar_TritonGPUToLLVM
  ElementwiseOpToLLVM.cpp
  ReduceOpToLLVM.cpp
  TypeConverter.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
