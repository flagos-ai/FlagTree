# Copyright 2026- Xcoresigma Technology Co., Ltd

add_triton_library(TleDSAToHIVM
  TleDistributedOpToHIVMConverter.cpp

  DEPENDS
  TleDSATableGen

  LINK_LIBS PUBLIC
  TleIR
  TleDSAIR
  MLIRIR
  MLIRTensorDialect
)
