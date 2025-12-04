add_triton_library(ZTCAnalysisStructured
  PtrAnalysis.cpp

  DEPENDS
  TritonAnalysis
  TritonTableGen
  TritonStructuredTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonStructuredIR
  MLIRAnalysis
)
