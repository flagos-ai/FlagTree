add_mlir_conversion_library(MLIRTritonToGCU_${arch}
  TritonToGCU.cpp
  ReduceOpToGCU.cpp
  ScanOpToGCU.cpp
  ElementwiseFusionOpToGCU.cpp
  MakeRangeOpToGCU.cpp
  LoadStoreOpToGCU.cpp
  TritonFuncOpFlatten.cpp
  TritonFusion.cpp
  TritonLoadStoreToDma.cpp
  TritonGCULayoutOptimize.cpp
  OptimizeDotLayout.cpp
  TritonConvertTensorPointer.cpp
  Utility.cpp
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
  )
