set(LLVM_TARGET_DEFINITIONS Canonicalize.td)
mlir_tablegen(TritonCanonicalize.inc -gen-rewriters)
add_public_tablegen_target(TritonCanonicalizeIncGen)

add_triton_library(FlagTree_sunrise_TritonIR
  Dialect.cpp
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
)
