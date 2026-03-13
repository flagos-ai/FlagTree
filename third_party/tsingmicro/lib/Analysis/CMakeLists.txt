add_triton_library(ZTCAnalysis
  Allocation.cpp
  Alias.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
