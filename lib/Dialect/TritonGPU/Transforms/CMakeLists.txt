set(TritonGPUTransformsTleSources)
if(FLAGTREE_TLE)
  list(APPEND TritonGPUTransformsTleSources
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/lib/Transforms/EncodingRematerialization.cpp
    Pipeliner/TleWGMMAAnalysis.cpp
    Pipeliner/TleWGMMAUserPromisePipeline.cpp
  )
endif()

# The pass needs ttg.concat_dot_operand, which tablegen only emits when
# __FLAGTREE_CONCAT_DOT_OPERAND__ is defined (see cmake/FlagTreeOptions.cmake).
set(TritonGPUTransformsConcatDotOperandSources)
if(NOT FLAGTREE_BACKEND)
  list(APPEND TritonGPUTransformsConcatDotOperandSources
    ConcatDotOperand.cpp
  )
endif()

add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  ${TritonGPUTransformsConcatDotOperandSources}
  ProcessSharedMemoryHint.cpp
  Coalesce.cpp
  F32DotTC.cpp
  FuseNestedLoops.cpp
  CombineTensorSelectAndIf.cpp
  DecomposeScaledBlocked.cpp
  HoistTMEMAlloc.cpp
  ReduceDataDuplication.cpp
  OptimizeAccumulatorInit.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/AssignLatencies.cpp
  Pipeliner/LowerLoops.cpp
  Pipeliner/MMAv5PipelineUtility.cpp
  Pipeliner/ScheduleLoops.cpp
  ${TritonGPUTransformsTleSources}
  Pipeliner/WGMMAPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/TestPipelineLowerLoop.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Pipeliner/TMAStoresPipeline.cpp
  Pipeliner/MMAv5PipelineUtility.cpp
  Pipeliner/PipeliningUtility.cpp
  Pipeliner/Schedule.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  CoalesceAsyncCopy.cpp
  Utility.cpp
  CoalesceUtils.cpp
  LayoutPropagationUtility.cpp
  WarpSpecialization/AutomaticWarpSpecialization.cpp
  WarpSpecialization/Partition.cpp
  WarpSpecialization/OptimizePartitionWarps.cpp
  WarpSpecialization/PartitionBuilder.cpp
  WarpSpecialization/PartitionLoops.cpp
  WarpSpecialization/PartitionScheduling.cpp

  DEPENDS
  TritonGPUTransformsIncGen
  # flagtree: fix missing Triton dependency
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  NVWSTransforms
  TritonToTritonGPU
  TritonInstrumentIR
  MLIRTransformUtils
)
