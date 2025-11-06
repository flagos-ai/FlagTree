add_triton_library(FlagTree_iluvatar_TritonDialectTritonGPUTransforms
  Prefetch.cpp
  ReduceDataDuplication.cpp
  RemoveLayoutConversions.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
