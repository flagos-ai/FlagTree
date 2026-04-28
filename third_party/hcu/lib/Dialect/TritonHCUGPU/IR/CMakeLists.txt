add_triton_library(TritonHCUGPUIR
  Dialect.cpp

  DEPENDS
  TritonHCUGPUTableGen
  TritonHCUGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
