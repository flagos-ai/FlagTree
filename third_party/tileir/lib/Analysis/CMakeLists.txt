set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(TritonTileIRAnalysis
  TkoAliasAnalysis.cpp
  TkoDependenceAnalysis.cpp
  TkoOpSemantics.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSCFDialect
  MLIRGPUDialect
  TritonIR
  ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
)
