set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonDialectTritonIR)
if(NOT TARGET ${_EXTRA_LINK_LIBS})
  set(_EXTRA_LINK_LIBS "")
endif()

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
