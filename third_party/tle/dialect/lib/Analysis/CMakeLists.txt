add_triton_library(TritonTLEAnalysis
  AxisInfoExt.cpp

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonGPUIR
  TritonIR
  TleIR
)
