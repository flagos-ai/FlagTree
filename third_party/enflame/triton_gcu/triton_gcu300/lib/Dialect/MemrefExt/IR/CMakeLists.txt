add_mlir_dialect_library(MemrefExtIR${arch}
  MemrefExtDialect.cpp
  MemrefExtOps.cpp

  DEPENDS
  MemrefExtTableGen${arch}

  LINK_LIBS PUBLIC
  MLIRIR
)
