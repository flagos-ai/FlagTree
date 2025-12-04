add_triton_library(ZTCAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  UseAnalysis.cpp

  DEPENDS
  TritonAnalysis
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
