
add_triton_library(TritonMETAXGPUTransforms
  AccelerateMETAXMatmul.cpp
  MACACommon.cpp
  PipelineMACABasic.cpp
  PipelineAsyncBase.cpp
  PipelineAsyncTN.cpp
  PipelineAsyncTT.cpp
  ChangeLayoutFromRepNToElemN.cpp
  OptimizeCStorePass.cpp
  OptimizeSmemUsage.cpp
  ChangeLayoutForInt8.cpp
  ChangeLayoutForConstancyLoad.cpp
  AddPtrOpt.cpp
  ChangeTransOpGraph.cpp

  DEPENDS
  TritonMETAXGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonMETAXGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMETAXGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
