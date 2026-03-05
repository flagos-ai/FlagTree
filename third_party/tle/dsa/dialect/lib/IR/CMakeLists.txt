# Copyright (c) 2025 XCoreSigma Inc. All rights reserved.

add_triton_library(TleIR
  Dialect.cpp
  TleOps.cpp

  DEPENDS
  TleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)