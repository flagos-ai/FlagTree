add_triton_library(Tx81IR
  IR/Tx81Dialect.cpp
  IR/Tx81Ops.cpp

  DEPENDS
  Tx81TableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
