add_triton_library(LLVMXPUTransforms
  MfenceCheck.cpp

  DEPENDS
  LLVMXPUTransformsIncGen
  TritonXPUTableGen
  TritonXPUAttrDefsIncGen
)
