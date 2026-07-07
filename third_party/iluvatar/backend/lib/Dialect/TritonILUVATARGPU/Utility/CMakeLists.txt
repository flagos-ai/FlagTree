add_triton_library(TritonILUVATARUtils
  CommonUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
