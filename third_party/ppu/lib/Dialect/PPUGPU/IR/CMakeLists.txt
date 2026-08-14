add_triton_library(PPUGPUIR
  Dialect.cpp

  DEPENDS
  PPUGPUTableGen
  PPUGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
