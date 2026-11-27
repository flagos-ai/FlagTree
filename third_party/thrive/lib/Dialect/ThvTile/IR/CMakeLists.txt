add_triton_library(ThvTileIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ThvTileTableGen
  ThvTileTypesIncGen
  ThvTileAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRDestinationStyleOpInterface
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
