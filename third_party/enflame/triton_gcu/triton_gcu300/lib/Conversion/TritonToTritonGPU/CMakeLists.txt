add_mlir_conversion_library(MLIRGCUTritonToTritonGPU_${arch}
  GCUConvertTritonToTritonGPU.cpp
  GCUTritonGPUConversion.cpp

  DEPENDS
  TritonGCUTableGen_${arch}
  MLIRTritonGCUConversionPassIncGen_${arch}
  triton_gcu300

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
  )
