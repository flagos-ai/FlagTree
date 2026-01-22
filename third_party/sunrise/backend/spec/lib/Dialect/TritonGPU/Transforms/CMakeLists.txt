include_directories(${PROJECT_SOURCE_DIR}/lib/Dialect/TritonGPU/Transforms/Pipeliner)

add_triton_library(FlagTree_sunrise_TritonGPUTransforms
  Pipeliner/PipeliningUtility.cpp
  Coalesce.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
