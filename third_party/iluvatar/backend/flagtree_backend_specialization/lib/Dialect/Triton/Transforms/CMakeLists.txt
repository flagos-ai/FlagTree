include_directories(${PROJECT_BINARY_DIR}/third_party/iluvatar/lib/Dialect/Triton/Transforms)

add_triton_library(FlagTree_iluvatar_TritonTransforms
  Combine.cpp
  RewriteTensorPointer.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
  TritonCombineIncGen
)
