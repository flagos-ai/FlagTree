# flagtree tle
add_triton_library(TritonTLETransforms
  TleEarlyAssignMemorySpace.cpp
  TleAssignLocalPointersEncoding.cpp
  TleInsertLocalPointerBarriers.cpp
  TleLowerAsyncLoad.cpp
  TleLowerTmaCopy.cpp
  ConvertArgToMemDesc.cpp
  DSLRegionInline.cpp

  DEPENDS
  TritonTLETransformsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
)
