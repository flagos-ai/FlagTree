add_triton_library(TleTools FlagcxUtils.cpp

  DEPENDS
  TleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
