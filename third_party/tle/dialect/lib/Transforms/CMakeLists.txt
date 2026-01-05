# flagtree tle
add_triton_library(TritonTLETransforms
  TleLowerTmaCopy.cpp

  DEPENDS
  TritonTLETransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
)
