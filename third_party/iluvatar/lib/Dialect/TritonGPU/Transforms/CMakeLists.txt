if(FLAGTREE_BACKEND STREQUAL "iluvatar")
  set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonDialectTritonGPUTransforms)
  set(_EXTRA_LINK_CPP "")
else()
  set(_EXTRA_LINK_LIBS "TritonNvidiaGPUIR")
  set(_EXTRA_LINK_CPP "Pipeliner/TMAStoresPipeline.cpp")
endif()

add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  F32DotTC.cpp
  CombineTensorSelectAndIf.cpp
  ReduceDataDuplication.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/OuterLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/SoftwarePipeliner.cpp
  ${_EXTRA_LINK_CPP}
  Pipeliner/PipeliningUtility.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  ${_EXTRA_LINK_LIBS}
  MLIRTransformUtils
)
