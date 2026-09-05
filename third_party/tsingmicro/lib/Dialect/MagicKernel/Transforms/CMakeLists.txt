add_triton_library(MKTransforms
  MaterializeStridedLinalgInputsPass.cpp

  DEPENDS
  MagicKernelTableGen
  MKTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  TritonIR
  TritonTransforms
  ZTCAnalysis
)
