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
  TleLowerWGMMA.cpp
  TleLowerAsyncLoad.cpp
  TleLowerPipeToNvws.cpp
  TleLowerPipeToRDNA.cpp
  TleRestorePipeFunctionCalls.cpp
  TleSharedOffsetFunctionABI.cpp
  TleLowerBarriers.cpp
  TleAllocateNamedBarriers.cpp
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
  TleAddDistributedParams.cpp

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

# TleLowerPipeToRDNA emits AMD-dialect mbarrier ops. The AMD dialect only exists
# when the amd codegen backend is built, so gate the dependency (and the
# implementation body, via TLE_AMD_PIPE) on it. The pass/factory still compile on
# every backend (a no-op) so GEN_PASS_REGISTRATION stays resolvable.
if(TARGET TritonAMDGPUIR)
  target_link_libraries(TritonTLETransforms PUBLIC TritonAMDGPUIR)
  target_compile_definitions(TritonTLETransforms PRIVATE TLE_AMD_PIPE)
endif()
