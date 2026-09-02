set(LLVM_TARGET_DEFINITIONS Canonicalize.td)
mlir_tablegen(TritonCanonicalize.inc -gen-rewriters)
add_public_tablegen_target(TritonCanonicalizeIncGen)

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
  # FlagTree sources include TritonGPU declarations from TritonIR.
  TritonGPUCGAAttrIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUOpsEnumsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
