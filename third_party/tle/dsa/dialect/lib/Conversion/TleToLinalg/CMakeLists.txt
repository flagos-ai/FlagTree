# Copyright 2026- Xcoresigma Technology Co., Ltd

add_triton_library(TleDSAToLinalg
  DSACopyConverter.cpp
  MathConverter.cpp

  DEPENDS
  TritonIR
  TleDSATableGen

  LINK_LIBS PUBLIC
  TleIR
  TritonIR
  MLIRIR
  MLIRMemRefDialect
)
