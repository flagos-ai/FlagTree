add_mlir_conversion_library(MLIRGCUTritonToTritonGPU_${arch}
  GCUConvertTritonToTritonGPU.cpp
  GCUTritonGPUConversion.cpp

  DEPENDS
  TritonGCUTableGen_${arch}
  MLIRTritonGCUConversionPassIncGen_${arch}
  triton_${arch}

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
  )
