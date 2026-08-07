if(ENABLE_TLE)
  set(_TLE_IR_LIBS TleIR)
else()
  set(_TLE_IR_LIBS "")
endif()

add_mlir_conversion_library(MLIRGCUTritonToTritonGPU_${arch}
  GCUConvertTritonToTritonGPU.cpp
  GCUTritonGPUConversion.cpp

  DEPENDS
  TritonGCUTableGen_${arch}
  MLIRTritonGCUConversionPassIncGen_${arch}
  triton_${arch}

  LINK_LIBS PUBLIC
  ${_TLE_IR_LIBS}
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRUBDialect
  )
