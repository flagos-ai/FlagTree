add_triton_library(FlagTree_sunrise_TritonTransforms
  RewriteTensorPointer.cpp

  DEPENDS
  TritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  TritonIR
)
