add_triton_library(TritonXPUTransforms
  Alloca.cpp
  AsyncLoadSchedule.cpp
  CoreTiling.cpp
  CreateGM2LM.cpp
  DtypeConvert.cpp
  Legalize.cpp
  TLELegalize.cpp
  LoopGrid.cpp
  LoopInvariantStaging.cpp
  LowerPrint.cpp
  Mask.cpp
  OffsetAnalysis.cpp
  ScalarAnalysis.cpp
  TileAnalysisPass.cpp
  VectorizabilityAnalysisPass.cpp
  Normalize.cpp
  Vectorize.cpp
  MemoryAsync.cpp
  UnrollControl.cpp
  Interleave.cpp
  StoreControl.cpp
  OtherSim.cpp
  MemoryCache.cpp
  MemoryInplace.cpp
  FuncConvert.cpp
  CFToSCF.cpp
  LegalizeExternEW.cpp

  DEPENDS
  TritonXPUTransformsIncGen
  TritonXPUTableGen
  TritonXPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonXPUAnalysis
  MLIRControlFlowToSCF
)

target_include_directories(TritonXPUTransforms BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xpu/include
  ${PROJECT_BINARY_DIR}/third_party/xpu/include
)
