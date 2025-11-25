get_flagtree_backend_lib("TritonIR" _EXTRA_LINK_LIBS)

add_triton_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  ${_EXTRA_LINK_LIBS}
)
