add_triton_library(FlagTree_ascend_TritonIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
