add_triton_library(MUSATLETransforms
  InsertLocalPointerBarriers.cpp
  LowerBarrierAllocations.cpp
  LowerBarrierOperations.cpp
  LowerExclusiveCumsum.cpp
  LowerPipe.cpp
  LowerSqmma.cpp
  LowerTMETransactions.cpp
  LowerWarpSpecialize.cpp
  LowerWarpSpecializeToLLVM.cpp
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
  TleIR
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
)
