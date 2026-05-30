set(LLVM_TARGET_DEFINITIONS Canonicalize.td)
mlir_tablegen(TritonCanonicalize.inc -gen-rewriters)
add_public_tablegen_target(TritonCanonicalizeIncGen)

get_flagtree_backend_lib("TritonIR" _EXTRA_LINK_LIBS)

add_triton_library(TritonIR
  Dialect.cpp
  DiscardableAttributes.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp
  OpInterfaces.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonCanonicalizeIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  ${_EXTRA_LINK_LIBS}
)
