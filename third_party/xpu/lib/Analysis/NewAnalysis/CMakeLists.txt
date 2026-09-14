add_library(TritonXPUAnalysisSDNN STATIC IMPORTED)
set_target_properties(TritonXPUAnalysisSDNN PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/SDNN/libTritonXPUAnalysisSDNN.a
)

add_mlir_library(TritonXPUAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  UseAnalysis.cpp
  Utility.cpp
  ../ScalarAnalysis.cpp
  ../TileAnalysis.cpp
  ../TileDecision.cpp
  ../VectorizabilityAnalysis.cpp

  DEPENDS
  # Dialect.h includes the XPU attribute-interface declarations.
  TritonXPUAttrDefsIncGen
  TritonXPUTableGen
)

target_link_libraries(TritonXPUAnalysis PRIVATE TritonXPUAnalysisSDNN)
