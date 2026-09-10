set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(TritonToTileIR
    MapElementwiseExpansion.cpp
    TritonToTileIRPass.cpp
    Utils.cpp

    DEPENDS
    TritonToTileIRConversionPassIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    # order is important
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonAnalysis
    TritonIR
    Utils
    ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
    ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileTransforms.a
)
