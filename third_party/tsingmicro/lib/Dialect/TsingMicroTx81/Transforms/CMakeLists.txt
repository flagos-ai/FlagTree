add_triton_library(Tx81Transforms
  InsertBarrierPass.cpp
  ResolveDmaBaseAddrPass.cpp

  DEPENDS
  Tx81TableGen
  Tx81TransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  TritonIR
  TritonTransforms
  ZTCAnalysis
  Tx81IR
)
