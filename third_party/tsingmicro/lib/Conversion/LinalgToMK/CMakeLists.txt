add_triton_library(LinalgToMagicKernel
  LinalgToMK.cpp
  LinalgToMKPass.cpp

  DEPENDS
  MagicKernelTableGen
  LinalgToMKConversionPassIncGen
  TritonUtils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonUtils
)
