add_triton_library(TritonXPUToLLVM
  AllocateXPUSharedMemory.cpp
  ConvertLayoutOpToLLVM.cpp
  ElementwiseOpToLLVM.cpp
  FuncOpToLLVM.cpp
  GPUOpToLLVMXPU.cpp
  LoadStoreOpToLLVM.cpp
  MakeRangeOpToLLVM.cpp
  ReduceOpToLLVM.cpp
  ScanOpToLLVM.cpp
  SPMDOpToLLVM.cpp
  TritonXPUToLLVM.cpp
  VectorizedOpToLLVM.cpp
  ViewOpToLLVM.cpp
  XPUPrintOpToLLVM.cpp
  XPUUtilityOpToLLVM.cpp

  TargetInfo.cpp
  TypeConverter.cpp
  Utility.cpp

  DEPENDS
  TTX2LLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonXPUAnalysis
  TritonGPUToLLVM
)
