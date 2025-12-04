#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(ConvertTritonPtr
  TritonPtrToMemrefPass.cpp
  ReconcilePtrCastsPass.cpp
  TritonPtrToAddressPass.cpp

  DEPENDS
  ConvertTritonPtrPassIncGen

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
