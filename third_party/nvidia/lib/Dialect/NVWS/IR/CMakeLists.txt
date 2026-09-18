add_triton_library(NVWSIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  NVWSTableGen
  NVWSAttrDefsIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
