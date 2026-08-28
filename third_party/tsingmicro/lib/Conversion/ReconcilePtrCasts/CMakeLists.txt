add_triton_library(TsmcReconcilePtrCasts
  ReconcilePtrCastsPass.cpp

  DEPENDS
  TsmcReconcilePtrCastsPassIncGen

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
