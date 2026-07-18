add_xpu_sdnn_object(TritonSDNNToLLVM
  ${CMAKE_CURRENT_SOURCE_DIR}/TritonSDNNToLLVM.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/TritonSDNNToLLVMPass.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/Utility.cpp.o

  LINK_LIBS
  MLIRIR
  MLIRPass
  TritonXPUAnalysis
  TritonIR
  TritonSDNNIR
  TritonSDNNTransforms
  MLIRAffineToStandard
)
