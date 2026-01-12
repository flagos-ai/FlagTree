add_triton_library(AllocateSharedMemory
  AllocateSharedMemoryPass.cpp

  DEPENDS
  AllocateSharedMemoryPassIncGen
  TritonUtils
  ZTCAnalysis

  LINK_LIBS PUBLIC
  TritonUtils
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  ZTCAnalysis
)
