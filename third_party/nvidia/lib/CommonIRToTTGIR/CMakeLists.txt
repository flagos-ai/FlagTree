add_triton_library(CommonIRToTTGIR
  CommonIRToTTGIR.cpp

  DEPENDS
  CommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  CommonIRIR
  TritonIR
  TritonGPUIR
)
