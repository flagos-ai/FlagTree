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
  FuncConvert.cpp
  CFToSCF.cpp

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
