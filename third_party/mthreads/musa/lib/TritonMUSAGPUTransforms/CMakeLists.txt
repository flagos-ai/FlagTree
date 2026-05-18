if(FLAGTREE_MTHREADS_TLE)
  set(_TLE_SOURCES
    TLE/LowerAsyncLoad.cpp
  )
else()
  set(_TLE_SOURCES "")
endif()

add_triton_library(TritonMUSAGPUTransforms
  AccelerateMUSAMatmul.cpp
  CanonicalizeSqmmaResultConversions.cpp
  ConvertSqmmaToMTGPU.cpp
  FinalizeBarriers.cpp
  IssueBarrierInsertion.cpp
  OptimizeDotOperands.cpp
  MarkInplaceLoads.cpp
  Pipeline.cpp
  OptimizeAccumulatorInit.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeSqmmaAccumulatorLayout.cpp
  SqmmaPipelineUtils.cpp
  TMEPipelineUtils.cpp
  TMELowering.cpp
  ${_TLE_SOURCES}

  DEPENDS
  TritonMUSAGPUTransformsIncGen
  MTGPUTableGen
  MTGPUTypesIncGen
  TritonGPUIR

  LINK_LIBS PUBLIC
  MTGPUIR
  MUSAIR
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
)

target_include_directories(TritonMUSAGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMUSAGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
