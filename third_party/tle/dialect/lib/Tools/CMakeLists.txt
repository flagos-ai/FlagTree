add_triton_library(TleTools FlagcxUtils.cpp

  DEPENDS
  TleTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  ${TLE_TRITON_GPU_ATTR_DEPENDS}
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
