#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(TritonToCoreDialects
  TritonToCoreDialectsPass.cpp

  DEPENDS
  TritonToCoreDialectsConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonAnalysis
  TritonIR
  TritonTransforms
  ZTCAnalysis

  TritonArithToLinalg
  StructuredToMemref
  UnstructuredToMemref
  TritonToStructured
  TritonToUnstructured
  ConvertTritonPtr
)
