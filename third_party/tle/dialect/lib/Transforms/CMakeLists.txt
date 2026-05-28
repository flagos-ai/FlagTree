# flagtree tle
add_triton_library(TritonTLETransforms
  TleEarlyAssignMemorySpace.cpp
  TleSelectEncodings.cpp
  TleInsertLocalPointerBarriers.cpp
  TleOptimizeLocalPointerLoads.cpp
  TleOptimizeLocalPointerStores.cpp
  TleOptimizeLocalPointerAsyncStores.cpp
  TlePromoteLocalStoreStaging.cpp
  TleTileStylePipelineSchedule.cpp
  TleMaterializeTileStylePipeline.cpp
  TleDowngradeInvalidAsyncCopy.cpp
  TleOptimizeExclusiveCumsumLayouts.cpp
  TleLowerExclusiveCumsum.cpp
  TleLowerAsyncLoad.cpp
  TleLowerPipeToNvws.cpp
  TleTileToLLVMUtils.cpp
  ExtractTileToLLVM.cpp
  InsertTileToLLVM.cpp
  TleLowerExtractTile.cpp
  TleLowerInsertTile.cpp
  TleLowerTmaCopy.cpp
  TleScheduleTmaStoreSync.cpp
  ConvertArgToMemDesc.cpp
  RemoveRedundantCopy.cpp
  DSLRegionInline.cpp
  TleUtility.cpp

  DEPENDS
  TritonTLETransformsIncGen

  LINK_LIBS PUBLIC
  TritonTLEAnalysis
  TritonGPUIR
  TritonAnalysis
  TritonIR
  NVWSIR
)

target_include_directories(TritonTLETransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/include
)
