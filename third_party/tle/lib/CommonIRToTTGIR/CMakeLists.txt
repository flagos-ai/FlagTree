add_triton_library(TleCommonIRToTTGIR
  CommonIRToTTGIR.cpp

  DEPENDS
  TleCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  CommonIRIR
  TritonIR
  TritonGPUIR
)
