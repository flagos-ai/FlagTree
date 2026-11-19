add_triton_library(HCUTLERawTransforms
  MaterializeDeferredRaw.cpp
  DeferredRawSourceRegistry.cpp

  DEPENDS
  HCUTLERawTransformsIncGen
  TleTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TleIR
  TritonIR
  TritonTLERawUtils
)

target_include_directories(HCUTLERawTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/hcu/tle_raw/include
  ${CMAKE_SOURCE_DIR}/third_party/tle/utils/include
  ${CMAKE_SOURCE_DIR}/third_party
)
