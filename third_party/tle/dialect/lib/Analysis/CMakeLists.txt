add_triton_library(TritonTLEAnalysis
  AxisInfoExt.cpp
  TleMemoryEffectAnalysis.cpp
  TlePipeEffectAnalysis.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  ${TLE_TRITON_GPU_ATTR_DEPENDS}
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonGPUIR
  TritonIR
  TleIR
)
