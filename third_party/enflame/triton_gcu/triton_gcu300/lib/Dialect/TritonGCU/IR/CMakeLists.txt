add_mlir_dialect_library(TritonGCUIR_${arch}
  TritonGCUDialect.cpp
  TritonGCUTypes.cpp
  TritonGCUOps.cpp

  DEPENDS
  TritonGCUTableGen_${arch}
  triton_gcu300

  LINK_LIBS PUBLIC
  MLIRIR
)
