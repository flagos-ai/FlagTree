add_triton_library(TritonTLERawUtils
  lib/Protocol.cpp
  lib/AnalyzeReturnType.cpp
  lib/TleRawMaterialize.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  ${TLE_TRITON_GPU_ATTR_DEPENDS}
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TleIR
  TritonIR
  MLIRTargetLLVMIRImport
)

target_include_directories(TritonTLERawUtils PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/tle/utils/include
  ${CMAKE_SOURCE_DIR}/python/src
)

# Match the TLE definitions used to generate the dialect and interface headers.
# Undefining __TLE__ here hides types referenced by those generated headers.
