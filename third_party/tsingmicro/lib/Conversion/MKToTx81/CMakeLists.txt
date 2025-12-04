add_triton_library(MKToTx81
  MKToTx81.cpp
  MKToTx81Pass.cpp

  DEPENDS
  Tx81TableGen
  MKToTx81ConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
