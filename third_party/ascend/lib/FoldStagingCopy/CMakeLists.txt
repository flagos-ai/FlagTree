add_triton_library(FoldStagingCopy
  FoldStagingCopy.cpp

  DEPENDS
  FoldStagingCopyConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRTensorDialect
)
