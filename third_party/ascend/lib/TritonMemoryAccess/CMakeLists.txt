add_triton_library(TritonMemoryAccessCore
  OpFoldResultUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
)

add_triton_library(TritonMemoryAccessAnalysis
  MaskAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
  TritonMemoryAccessCore
)
