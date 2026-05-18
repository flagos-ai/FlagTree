add_triton_library(TritonMUSAGPUToLLVM
  AllocateSharedMemory.cpp
  BarrierOpToLLVM.cpp
  ConvertLayoutOpToLLVM.cpp
  DotOpToLLVM.cpp
  DotOpToLLVM/SQMMA.cpp
  DotOpToLLVM/WMMA.cpp
  ElementwiseOpToLLVM.cpp
  LoadStoreOpToLLVM.cpp
  MUSAOpsToLLVM.cpp
  SPMDOpToLLVM.cpp
  TensorPtrOpsToLLVM.cpp
  TritonGPUToLLVM.cpp
  TargetInfo.cpp
  ThreadIdOpToLLVM.cpp
  Utility.cpp
  WarpIdOpToLLVM.cpp

  DEPENDS
  TritonMUSAGPUConversionPassIncGen
  MUSAAttrDefsIncGen
  MTGPUTableGen
  MTGPUTypesIncGen

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonGPUToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRUBToLLVM
  MTGPUIR
  MUSAIR
  MLIRGPUToMTGPUTransforms
)
