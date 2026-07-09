add_triton_library(MTGPUToLLVM
  MTGPUToLLVMPass.cpp

  DEPENDS
  MTGPUConversionPassIncGen
  MTGPUTableGen
  MTGPUTypesIncGen

  LINK_LIBS PUBLIC
  TritonMUSAGPUToLLVM
  MTGPUIR
  MLIRControlFlowDialect
  MLIRFuncTransforms
)
