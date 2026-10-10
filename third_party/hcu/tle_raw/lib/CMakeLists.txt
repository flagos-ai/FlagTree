add_triton_library(HcuTLERawTransforms
  MaterializeDeferredRaw.cpp
  DeferredRawSourceRegistry.cpp

  DEPENDS
  HcuTLERawTransformsIncGen
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

target_include_directories(HcuTLERawTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/hcu/tle_raw/include
  ${CMAKE_SOURCE_DIR}/third_party/tle/utils/include
  ${CMAKE_SOURCE_DIR}/third_party
)
