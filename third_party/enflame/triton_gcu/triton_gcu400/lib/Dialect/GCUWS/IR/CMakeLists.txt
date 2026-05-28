include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mlir_conversion_library(GCUWSIR_${arch}
  Dialect.cpp
  Ops.cpp

  DEPENDS
  GCUWSTableGen_${arch}
  GCUWSAttrDefsIncGen_${arch}
  triton_gcu400

  LINK_LIBS PUBLIC
  MLIRIR
)
