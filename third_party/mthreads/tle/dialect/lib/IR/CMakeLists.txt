add_triton_library(MUSATLEIR
  Dialect.cpp

  DEPENDS
  MUSATLETableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
