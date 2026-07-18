add_triton_library(McTleIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  McTleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
