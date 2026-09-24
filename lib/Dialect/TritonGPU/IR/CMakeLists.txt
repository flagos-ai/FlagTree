add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  WarpSlice.cpp
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

add_dependencies(TritonGPUIR TritonGPUCTAAttrIncGen)
