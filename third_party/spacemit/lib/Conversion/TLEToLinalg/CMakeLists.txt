add_triton_library(TLEToLinalg
  TLEToLinalg.cpp
  TLEToLinalgPass.cpp

  DEPENDS
  TLEToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TleIR
)
