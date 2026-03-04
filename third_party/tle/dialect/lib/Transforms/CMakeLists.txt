# flagtree tle
add_triton_library(TritonTLETransforms
  TleEarlyAssignMemorySpace.cpp
  TleAssignLocalPointersEncoding.cpp
  TleInsertLocalPointerBarriers.cpp
  TleLowerAsyncLoad.cpp
  ExtractTileToLLVM.cpp
  TleLowerExtractTile.cpp
  TleLowerTmaCopy.cpp
  ConvertArgToMemDesc.cpp
  DSLRegionInline.cpp

  DEPENDS
  TritonTLETransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TritonAnalysis
)

target_include_directories(TritonTLETransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/include  # 确保能找到 LinearLayoutConversions.h
)
