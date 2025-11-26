add_mlir_dialect_library(MathExtIR${arch}
  MathExtDialect.cpp
  MathExtOps.cpp
  MathExtTypes.cpp

  DEPENDS
  MathExtTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
