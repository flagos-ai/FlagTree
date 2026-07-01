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
)

target_link_libraries(TritonXPUAnalysis PRIVATE TritonXPUAnalysisSDNN)
