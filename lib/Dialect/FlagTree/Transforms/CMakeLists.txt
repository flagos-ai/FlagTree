add_triton_library(FlagTreeTransforms
  ConvertArgToMemDesc.cpp
  DSLRegionInline.cpp

  DEPENDS
  TritonIR
  TritonGPUIR
  GluonIR
  FlagTreeIR
  FlagTreeTransformsIncGen
)
