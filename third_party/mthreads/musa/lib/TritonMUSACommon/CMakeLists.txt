add_triton_library(TritonMUSACommon
  ConvertLayoutUtils.cpp

  DEPENDS
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonGPUIR
  TritonIR
)

target_include_directories(TritonMUSACommon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMUSACommon PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
