add_triton_library(AllocateSharedMemory
  AllocateSharedMemoryPass.cpp

  DEPENDS
  AllocateSharedMemoryPassIncGen
  TritonSharedUtils
  TritonSharedAnalysis
  ZTCAnalysis

  LINK_LIBS PUBLIC
  TritonSharedUtils
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonSharedAnalysis
  ZTCAnalysis
)
