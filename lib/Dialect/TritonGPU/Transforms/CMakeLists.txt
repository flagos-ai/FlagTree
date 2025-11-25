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
  ${_TMA_LINK_CPP}
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
  ${_NVIDIA_LINK_LIBS}
  ${_EXTRA_LINK_LIBS}
  MLIRTransformUtils
)
