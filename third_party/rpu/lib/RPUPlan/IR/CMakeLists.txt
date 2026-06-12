add_triton_library(RPUPlanIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  RPUPlanTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
