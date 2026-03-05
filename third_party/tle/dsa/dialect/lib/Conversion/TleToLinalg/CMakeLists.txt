# Copyright (c) 2025 XCoreSigma Inc. All rights reserved.

add_triton_library(TleToLinalg
  DSACopyConverter.cpp
  MathConverter.cpp

  DEPENDS
  TritonIR
  TleTableGen
)