add_triton_library(TritonTLEAnalysis
  AxisInfoExt.cpp
  TleMemoryEffectAnalysis.cpp
  TlePipeEffectAnalysis.cpp

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonGPUIR
  TritonIR
  TleIR
)
