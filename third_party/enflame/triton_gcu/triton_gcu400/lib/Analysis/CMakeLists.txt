add_mlir_library(TritonGCUAnalysis_${arch}
  AxisInfoEx.cpp
  FirstLastUserAnalysis.cpp
  MaskAnalysis.cpp
  PtrAnalysis.cpp
  OpFoldResultUtils.cpp
  TestFirstLastUserAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/triton_gcu/Analysis/

  DEPENDS
  TritonGCUTableGen_${arch}
  triton_gcu400

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
