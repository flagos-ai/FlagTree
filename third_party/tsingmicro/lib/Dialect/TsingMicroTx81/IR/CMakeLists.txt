add_triton_library(Tx81IR
  Tx81Dialect.cpp
  Tx81Ops.cpp

  DEPENDS
  Tx81TableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
