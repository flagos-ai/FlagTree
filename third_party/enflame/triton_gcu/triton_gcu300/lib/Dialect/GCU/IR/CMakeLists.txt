add_mlir_dialect_library(GCUIR${arch}
  Dialect.cpp
  Types.cpp
  Ops.cpp

  DEPENDS
  GCUTableGen${arch}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
