include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_TLE)
  set(_TLE_IR_LIBS TleIR)
else()
  set(_TLE_IR_LIBS "")
endif()

add_mlir_conversion_library(MLIRTritonToGCU_${arch}
  TritonToGCU.cpp
  ReduceOpToGCU.cpp
  ElementwiseFusionOpToGCU.cpp
  ScanOpToGCU.cpp
  TritonGCUToGCU/TritonGCUToGCUUtils.cpp
  TritonFuncOpFlatten.cpp
  TritonFusion.cpp
  TritonLoadStoreToDma.cpp
  TritonGCULayoutOptimize.cpp
  OptimizeDotLayout.cpp
  TritonConvertTensorPointer.cpp
  Utils.cpp
  TritonGCUPingpong/TritonGCUPingpong.cpp
  TritonGCUPingpong/MatmulLoopPipeline.cpp
  TritonGCUPingpong/PipelineExpander.cpp
  TritonGCUPingpong/PipeliningUtility.cpp
  TritonGPUtoTrtionGCU.cpp
  TleToTritonGCU.cpp
  
  DEPENDS
  TritonGCUTableGen_${arch}
  MLIRTritonGCUConversionPassIncGen_${arch}
  triton_gcu300

  MemrefExtIR_${arch}
  MathExtIR_${arch}

  LINK_LIBS PUBLIC
  TritonGCUAnalysis_${arch}
  MemrefExtIR_${arch}
  MathExtIR_${arch}
  ${_TLE_IR_LIBS}
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRComplexDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTritonGCUCommon_${arch}
  )

target_include_directories(MLIRTritonToGCU_${arch} PRIVATE ../Common)