set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(Utils
  Utils.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
  ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
)
