add_triton_library(IluvatarTLERawTransforms
  MaterializeDeferredRaw.cpp

  DEPENDS
  IluvatarTLERawTransformsIncGen
  IluvatarTleTableGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonIR
  IluvatarTleRawUtils
)

target_include_directories(IluvatarTLERawTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/iluvatar/tle/dialect/include
  ${CMAKE_BINARY_DIR}/third_party/iluvatar/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/iluvatar/tle_raw/include
  ${CMAKE_SOURCE_DIR}/third_party/iluvatar/tle/utils/include
)
