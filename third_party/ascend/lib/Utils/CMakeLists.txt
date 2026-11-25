add_triton_library(MLIRTritonNPUUtils
        Utils.cpp
        InterleaveOptimization.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
  TritonMemoryAccessAnalysis
  TritonMemoryAccessCore
)
