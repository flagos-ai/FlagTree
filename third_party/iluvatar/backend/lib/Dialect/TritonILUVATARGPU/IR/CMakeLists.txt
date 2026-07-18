add_triton_library(TritonILUVATARGPUIR
  Dialect.cpp

  DEPENDS
  TritonILUVATARGPUTableGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
