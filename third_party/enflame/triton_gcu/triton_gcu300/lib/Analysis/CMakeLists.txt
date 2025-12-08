add_mlir_library(TritonGCUAnalysis_${arch}
  AxisInfoEx.cpp
  FirstLastUserAnalysis.cpp
  MaskAnalysis.cpp
  PtrAnalysis.cpp
  OpFoldResultUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/triton_gcu/Analysis/

  DEPENDS
  TritonGCUTableGen_${arch}
  triton_${arch}

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
