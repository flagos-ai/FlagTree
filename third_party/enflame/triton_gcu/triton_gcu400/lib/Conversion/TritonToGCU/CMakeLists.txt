if(ENABLE_TLE)
  set(_TLE_IR_LIBS TleIR)
else()
  set(_TLE_IR_LIBS "")
endif()

add_mlir_conversion_library(MLIRTritonToGCU_${arch}
  TritonToGCU.cpp
  ReduceOpToGCU.cpp
  AccelerateMatmul.cpp
  OptimizeDotOperands.cpp
  TritonWGDotToGCU.cpp
  ScanOpToGCU.cpp
  ElementwiseFusionOpToGCU.cpp
  MakeRangeOpToGCU.cpp
  LoadStoreOpToGCU.cpp
  TritonFuncOpFlatten.cpp
  TritonFusion.cpp
  TritonLoadStoreToDma.cpp
  TritonGCURemoveLayoutConversions.cpp
  TritonGCULayoutOptimize.cpp
  TritonConvertTensorPointer.cpp
  Utility.cpp
  TritonGPUtoTrtionGCU.cpp
  TTSmemLowerToGCU.cpp
  TleLowerPipeToGcuws.cpp
  TleToTritonGCU.cpp
  TleOpToGCU.cpp
  TritonGCULocalMemOptimize.cpp
  TritonGCUInsertProducerFences.cpp
  AllocateWarpGroups.cpp
  AnnotateDotAccReuse.cpp
  AnnotateDotFusion.cpp
  AnnotateDotAllocaReuse.cpp
  ReduceScanCommon.cpp
  Vectorize.cpp
  Constants.h

  DEPENDS
  TritonGCUTableGen_${arch}
  MLIRTritonGCUConversionPassIncGen_${arch}
  # Several sources below (TritonToGCU.cpp, TTSmemLowerToGCU.cpp,
  # TleLowerPipeToGcuws.cpp, TritonGCUInsertProducerFences.cpp) and
  # Transforms/Passes.h include "Dialect/GCUWS/IR/Dialect.h", which pulls in
  # the generated Dialect.h.inc / Ops.h.inc / Types.h.inc /
  # GCUWSAttrDefs.h.inc / GCUWSAttrEnums.h.inc / GCUWSOpInterfaces.h.inc.
  # Declare the tablegen targets here as well, otherwise the generated
  # headers are only produced as a side effect of building GCUWSIR_${arch}
  # and the build order is not guaranteed (race -> "fatal error:
  # Dialect/GCUWS/IR/Dialect.h.inc: No such file or directory").
  GCUWSTableGen_${arch}
  GCUWSAttrDefsIncGen_${arch}
  triton_gcu400

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
  MLIRTensorDialect
  MLIRTransforms
  )
