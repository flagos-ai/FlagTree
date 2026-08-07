add_triton_library(TritonPPUGPUIR
  Dialect.cpp

  DEPENDS
  TritonPPUGPUTableGen
  TritonPPUGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
