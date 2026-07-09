add_triton_ut(
  NAME TestOptimizeLDS
  SRCS OptimizeLDSTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonHCUGPUToLLVM
    MLIRUBToLLVM
    TritonHCUUtils
    TritonHCUAnalysis
    TritonHCUGPUTransforms
    TritonHCUGPUDialectToLLVM
)
