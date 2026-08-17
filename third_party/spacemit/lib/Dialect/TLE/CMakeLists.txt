add_triton_library(TLEIR
  TLEOps.cpp

  DEPENDS
  TLETableGen

  LINK_LIBS PUBLIC
  TritonIR
)
