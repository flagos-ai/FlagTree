add_triton_library(MUSAIR
  Dialect.cpp

  DEPENDS
  MUSATableGen
  MUSAAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
