add_triton_library(TritonToHFusion
  TritonToHFusion.cpp

  DEPENDS
    TritonToHFusionConversionPassIncGen

  LINK_LIBS
    BiShengIRHFusionDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    TritonIR
)
