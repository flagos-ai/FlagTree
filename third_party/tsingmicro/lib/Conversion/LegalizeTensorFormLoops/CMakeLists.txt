#===------------------------------------------------------------------------===#
#
# Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
# All rights reserved.
#
#===------------------------------------------------------------------------===#

add_triton_library(LegalizeTensorFormLoops
  LegalizeTensorFormLoops.cpp

  DEPENDS
  LegalizeTensorFormLoopsPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
)
