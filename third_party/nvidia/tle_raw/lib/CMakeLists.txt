add_triton_library(NvidiaTLERawTransforms
  MaterializeDeferredRaw.cpp
  DeferredRawSourceRegistry.cpp

  DEPENDS
  NvidiaTLERawTransformsIncGen

  LINK_LIBS PUBLIC
  TleIR
  TritonIR
  TritonTLERawUtils
)

target_include_directories(NvidiaTLERawTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/nvidia/tle_raw/include
  ${CMAKE_SOURCE_DIR}/third_party/tle/utils/include
)
