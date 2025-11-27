include_directories(${PROJECT_BINARY_DIR}/lib)

add_triton_library(FlagTree_iluvatar_TritonTransforms
  Combine.cpp
  RewriteTensorPointer.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
  TritonCombineIncGen
)
