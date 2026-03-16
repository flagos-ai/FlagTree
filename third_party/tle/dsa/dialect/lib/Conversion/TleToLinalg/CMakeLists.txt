# Copyright 2026- Xcoresigma Technology Co., Ltd

add_triton_library(TleToLinalg
  DSACopyConverter.cpp
  MathConverter.cpp

  DEPENDS
  TritonIR
  TleTableGen
)
