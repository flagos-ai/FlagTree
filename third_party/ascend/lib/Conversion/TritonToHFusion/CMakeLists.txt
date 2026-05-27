add_triton_library(TritonToHFusion
  TritonToHFusion.cpp

  DEPENDS
    TritonToHFusionConversionPassIncGen
    TritonAscendTableGen
    TritonAscendAttrDefsIncGen

  LINK_LIBS
    BiShengIRHFusionDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    TritonIR
)
