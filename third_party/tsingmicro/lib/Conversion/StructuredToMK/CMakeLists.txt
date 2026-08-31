add_triton_library(StructuredToMK
  StructuredToMK.cpp
  StructuredToMKPass.cpp

  DEPENDS
  StructuredToMKConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
  MLIRAddress
)
