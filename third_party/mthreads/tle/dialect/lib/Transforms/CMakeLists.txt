add_triton_library(MUSATLETransforms
  InsertLocalPointerBarriers.cpp
  OptimizeLocalPointerAsyncStores.cpp
  OptimizeLocalPointerLoads.cpp
  OptimizeLocalPointerStores.cpp
  SelectEncodings.cpp

  DEPENDS
  MUSATLETableGen
  TritonMUSAGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MUSATLEIR
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
)
