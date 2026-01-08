if(IS_COMPILE_TritonNvidiaGPU)
  set(_TMA_LINK_CPP "Pipeliner/TMAStoresPipeline.cpp")
  set(_NVIDIA_LINK_LIBS "TritonNvidiaGPUIR")
else()
  set(_TMA_LINK_CPP "")
  set(_NVIDIA_LINK_LIBS "")
endif()
get_flagtree_backend_lib("TritonGPUTransforms" _EXTRA_LINK_LIBS)

add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  Canonicalize.cpp
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
  WarpSpecialization/AutomaticWarpSpecialization.cpp
  WarpSpecialization/LoadMMASpecialization.cpp
  WarpSpecialization/Partition.cpp
  WarpSpecialization/OptimizePartitionWarps.cpp
  WarpSpecialization/PartitionBuilder.cpp
  WarpSpecialization/PartitionLoops.cpp
  WarpSpecialization/PartitionScheduling.cpp
  WarpSpecialization/RewritePartitionDependencies.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonToTritonGPU
  MLIRTransformUtils
)
