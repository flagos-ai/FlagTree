add_triton_library(TritonGPUIR
  ${PROJECT_SOURCE_DIR}/lib/Dialect/TritonGPU/IR/Dialect.cpp
  ${PROJECT_SOURCE_DIR}/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
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
