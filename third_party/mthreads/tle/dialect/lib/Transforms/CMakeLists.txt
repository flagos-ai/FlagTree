add_triton_library(MUSATLETransforms
  InsertLocalPointerBarriers.cpp
  LowerBarrierAllocations.cpp
  LowerExclusiveCumsum.cpp
  OptimizeLocalPointerAsyncStores.cpp
  OptimizeLocalPointerLoads.cpp
  OptimizeLocalPointerStores.cpp
  SelectEncodings.cpp

  DEPENDS
  MUSATLETableGen
  TritonMUSAGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MUSATLEIR
  MUSAIR
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
)
