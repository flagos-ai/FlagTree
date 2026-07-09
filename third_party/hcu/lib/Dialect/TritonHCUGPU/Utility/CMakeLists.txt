add_triton_library(TritonHCUUtils
  CommonUtils.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
