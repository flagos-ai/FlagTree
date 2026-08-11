add_triton_library(TritonPtrToMemref
  TritonPtrToMemrefPass.cpp

  DEPENDS
  TritonPtrToMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
)
