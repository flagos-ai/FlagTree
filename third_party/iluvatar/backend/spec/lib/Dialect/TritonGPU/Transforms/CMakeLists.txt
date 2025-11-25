include_directories(${PROJECT_SOURCE_DIR}/lib/Dialect/TritonGPU/Transforms/Pipeliner)

add_triton_library(FlagTree_iluvatar_TritonGPUTransforms
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Prefetch.cpp
  ReduceDataDuplication.cpp
  RemoveLayoutConversions.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
