add_triton_library(MUSATLEFrontendTransforms
  EarlyAssignMemorySpace.cpp
  LowerAsyncLoad.cpp

  DEPENDS
  MUSATLEFrontendTransformsIncGen
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  MUSAIR

  LINK_LIBS PUBLIC
  MUSAIR
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
)

target_include_directories(MUSATLEFrontendTransforms
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
)
