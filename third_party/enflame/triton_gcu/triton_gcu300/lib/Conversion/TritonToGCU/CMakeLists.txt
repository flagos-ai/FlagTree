include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_mlir_conversion_library(MLIRTritonToGCU_${arch}
  TritonToGCU.cpp
  ReduceOpToGCU.cpp
  ElementwiseFusionOpToGCU.cpp
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
  TritonGCUPingpong/OuterLoopPipeline.cpp
  TritonGCUPingpong/PipelineExpander.cpp
  TritonGCUPingpong/PipeliningUtility.cpp
  TritonGPUtoTrtionGCU.cpp

  DEPENDS
  TritonGCUTableGen_${arch}
  MLIRTritonGCUConversionPassIncGen_${arch}
  triton_${arch}

  MemrefExtIR${arch}
  MathExtIR${arch}

  LINK_LIBS PUBLIC
  TritonGCUAnalysis_${arch}
  MemrefExtIR${arch}
  MathExtIR${arch}
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
