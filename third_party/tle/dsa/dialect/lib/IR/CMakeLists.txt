# Copyright 2026- Xcoresigma Technology Co., Ltd

add_triton_library(TleDSAIR
  Dialect.cpp
  TleOps.cpp

  DEPENDS
  TleTableGen
  TleDSATableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)
