add_triton_library(TLEToLinalg
  TLEToLinalg.cpp
  TLEToLinalgPass.cpp

  DEPENDS
  TLEToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TLEIR
)
