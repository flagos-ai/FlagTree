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

add_xpu_sdnn_object(FuncConvert
  ${CMAKE_CURRENT_SOURCE_DIR}/FuncConvert.o

  LINK_LIBS
  TritonAnalysis
)
