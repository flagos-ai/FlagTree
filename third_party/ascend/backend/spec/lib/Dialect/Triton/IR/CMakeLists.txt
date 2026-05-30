add_triton_library(FlagTree_ascend_TritonIR
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
)

target_include_directories(FlagTree_ascend_TritonIR PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/lib/Dialect/Triton/IR
)
