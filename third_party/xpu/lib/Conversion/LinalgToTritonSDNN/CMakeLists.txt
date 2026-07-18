add_xpu_sdnn_object(LinalgToTritonSDNN
  ${CMAKE_CURRENT_SOURCE_DIR}/LinalgToTritonSDNNPass.cpp.o

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRSCFDialect
)
