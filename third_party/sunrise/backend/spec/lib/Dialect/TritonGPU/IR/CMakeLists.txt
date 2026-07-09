add_triton_library(FlagTree_sunrise_TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
