add_triton_library(ReconcilePtrCasts
  ReconcilePtrCastsPass.cpp

  DEPENDS
  ReconcilePtrCastsPassIncGen

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
  MLIRAddress
)
