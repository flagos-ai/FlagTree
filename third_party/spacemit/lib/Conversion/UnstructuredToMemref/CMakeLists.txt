#===------------------------------------------------------------------------===#
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
#===------------------------------------------------------------------------===#

add_triton_library(UnstructuredToMemref
  UnstructuredToMemrefPass.cpp

  DEPENDS
  UnstructuredToMemrefConversionPassIncGen

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
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
)
