add_triton_library(TleCommonIRToTTGIR
  CommonIRToTTGIR.cpp

  DEPENDS
  TleCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransformUtils
  CommonIRIR
  TritonIR
  TritonGPUIR
)
