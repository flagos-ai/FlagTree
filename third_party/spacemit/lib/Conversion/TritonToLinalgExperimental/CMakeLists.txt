#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(TritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  ReconcilePtrCastsPass.cpp
  ReconcileLlvmPtrCastsPass.cpp
  TritonToPtrPass.cpp
  ScfbufferStandardizedPass.cpp
  ConvertScanOpPass.cpp
  #CollapseShape.cpp

  DEPENDS
  TritonToLinalgExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  TritonSharedUtils

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
  AddTargetDescription
  XSMTToLinalg
  TLEToLinalg
  ProtonIR
)
