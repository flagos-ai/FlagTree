# flagtree tle
add_triton_library(TritonTLETransforms
  TleLowerAsyncLoad.cpp
  TleLowerTmaCopy.cpp

  DEPENDS
  TritonTLETransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
)
