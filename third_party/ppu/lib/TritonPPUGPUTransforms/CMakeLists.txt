add_triton_library(TritonPPUGPUTransforms
  AcceleratePPUMatmul.cpp
  AIULowering.cpp
  TlePromoteAsyncLoadToAIU.cpp

  DEPENDS
  TritonPPUGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonPPUGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonPPUGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
