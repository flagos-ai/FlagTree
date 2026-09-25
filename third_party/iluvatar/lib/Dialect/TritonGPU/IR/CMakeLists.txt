flagtree_add_iluvatar_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonGPUCTAAttrIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
