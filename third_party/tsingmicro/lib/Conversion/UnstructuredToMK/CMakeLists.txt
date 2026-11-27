add_triton_library(UnstructuredToMK
  UnstructuredToMKPass.cpp

  DEPENDS
  UnstructuredToMKConversionPassIncGen

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
  MLIRAddress
)
