add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  TensorMemoryUtils.cpp
  Ops.cpp

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen
  TritonNvidiaGPUTypesIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCGAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
