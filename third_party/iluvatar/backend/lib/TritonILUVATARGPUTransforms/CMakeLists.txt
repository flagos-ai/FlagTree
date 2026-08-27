add_triton_library(TritonILUVATARGPUTransforms
  SmeLoad.cpp
  OptimizeEpilogue.cpp
  MMAReduceThreadLocality.cpp
  ChainDotKRotate.cpp
  FAPipeline.cpp
  UpdateAsyncWaitCount.cpp

  DEPENDS
  TritonILUVATARGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonILUVATARGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonILUVATARGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
