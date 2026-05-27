# set(LLVM_TARGET_DEFINITIONS Canonicalize.td)
# mlir_tablegen(TritonCanonicalize.inc -gen-rewriters)
# add_public_tablegen_target(TritonCanonicalizeIncGen)

add_triton_library(FlagTree_ascend_TritonIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp

  DEPENDS
  TritonTableGen
  TritonCanonicalizeIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  TritonIR
)

target_include_directories(FlagTree_ascend_TritonIR PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/lib/Dialect/Triton/IR
)
