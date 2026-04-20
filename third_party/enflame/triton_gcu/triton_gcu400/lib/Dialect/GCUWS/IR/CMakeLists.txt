include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mlir_conversion_library(GCUWSIR_${arch}
  Dialect.cpp
  Ops.cpp

  DEPENDS
  GCUWSTableGen_${arch}
  GCUWSAttrDefsIncGen_${arch}
  triton_${arch}

  LINK_LIBS PUBLIC
  MLIRIR
)

target_compile_options(obj.GCUWSIR_${arch}
  PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-error=comment -Wno-unused-variable -Wno-unused-but-set-variable>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-error=comment -Wno-unused-variable>
)
