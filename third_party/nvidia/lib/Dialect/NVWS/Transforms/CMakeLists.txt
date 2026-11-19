add_triton_library(NVWSTransforms
  LowerAref.cpp
  LowerWarpGroup.cpp
  InsertAref.cpp
  Utilities.cpp
  AssignStagePhase.cpp
  InsertTmemAref.cpp
  HoistTmemStore.cpp

  DEPENDS
  NVWSTransformsIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCGAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
