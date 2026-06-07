add_triton_library(TritonAscendIR
  TritonAscendAttrs.cpp
  TritonAscendDialect.cpp
  TritonAscendOps.cpp

  DEPENDS
  TritonAscendTableGen
  TritonAscendAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
)
