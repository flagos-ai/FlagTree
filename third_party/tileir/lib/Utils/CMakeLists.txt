set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(Utils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
  ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
)
