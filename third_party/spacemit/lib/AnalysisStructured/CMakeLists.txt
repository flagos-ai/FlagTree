add_triton_library(TritonSharedAnalysisStructured
  PtrAnalysis.cpp

  DEPENDS
  TritonTableGen
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  TritonStructuredIR
  MLIRAnalysis
)
