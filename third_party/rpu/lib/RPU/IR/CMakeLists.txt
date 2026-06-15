add_triton_library(RPUIR
  Dialect.cpp
  ExecutableKind.cpp
  Ops.cpp

  DEPENDS
  RPUTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
