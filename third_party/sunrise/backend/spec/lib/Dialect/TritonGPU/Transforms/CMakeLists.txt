add_triton_library(FlagTree_sunrise_TritonGPUTransforms
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
