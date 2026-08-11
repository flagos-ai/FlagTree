add_triton_library(XSMTAsyncIR
  XSMTAsync.cc

  DEPENDS
  XSMTAsyncTableGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
)
