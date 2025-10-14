add_triton_library(FlagTree_iluvatar_TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)