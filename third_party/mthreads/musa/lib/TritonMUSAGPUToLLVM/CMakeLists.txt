if(FLAGTREE_MTHREADS_TLE)
  set(_TLE_LIBS MUSATLEToLLVM)
  if(FLAGCX_ENABLED)
    # Slim mthreads FlagCX lowering (not the full shared TleToLLVM).
    list(APPEND _TLE_LIBS FlagCxToLLVM)
  endif()
else()
  set(_TLE_LIBS "")
endif()

add_triton_library(TritonMUSAGPUToLLVM
  AllocateSharedMemory.cpp
  BarrierOpToLLVM.cpp
  ConvertLayoutOpToLLVM.cpp
  DotOpToLLVM.cpp
  DotOpToLLVM/SQMMA.cpp
  DotOpToLLVM/WMMA.cpp
  ElementwiseOpToLLVM.cpp
  Fp4ToFpOpToLLVM.cpp
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
  TritonMUSACommon
  TritonAnalysis
  TritonGPUToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRUBToLLVM
  MTGPUIR
  MUSAIR
  ${_TLE_LIBS}
  MLIRGPUToMTVMTransforms
)
