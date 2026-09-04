add_triton_library(TleTools
  FlagcxUtils.cpp

  DEPENDS
  TleTableGen
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUOpInterfacesIncGen
)
