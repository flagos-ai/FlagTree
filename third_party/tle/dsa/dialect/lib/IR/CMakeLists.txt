# Copyright 2026- Xcoresigma Technology Co., Ltd

add_triton_library(TleIR
  Dialect.cpp
  TleOps.cpp

  DEPENDS
  TleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)
