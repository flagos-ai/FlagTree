add_triton_library(TritonXPUTransforms
  Alloca.cpp
  CoreTiling.cpp
  CreateGM2LM.cpp
  DtypeConvert.cpp
  Legalize.cpp
  LoopGrid.cpp
  LowerPrint.cpp
  Mask.cpp
  OffsetAnalysis.cpp
  Vectorize.cpp
  MemoryAsync.cpp
  UnrollControl.cpp
  Interleave.cpp
  StoreControl.cpp
  OtherSim.cpp
  MemoryCache.cpp
  MemoryInplace.cpp
  CFToSCF.cpp

  DEPENDS
  TritonXPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonAnalysis
)
