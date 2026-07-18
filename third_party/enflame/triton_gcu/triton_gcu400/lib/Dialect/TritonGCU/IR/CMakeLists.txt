add_mlir_dialect_library(TritonGCUIR_${arch}
  TritonGCUDialect.cpp
  TritonGCUTypes.cpp
  TritonGCUOps.cpp

  DEPENDS
  TritonGCUTableGen_${arch}
  triton_gcu400

  LINK_LIBS PUBLIC
  MLIRIR
)
