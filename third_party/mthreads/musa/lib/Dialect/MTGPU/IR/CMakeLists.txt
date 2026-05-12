add_triton_library(MTGPUIR
  Dialect.cpp

  DEPENDS
  MTGPUTableGen
  MTGPUTypesIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
