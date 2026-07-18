add_mlir_conversion_library(MLIRTritonGCUCommon_${arch}
  ConstantUtil.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSupport
  MLIRSideEffectInterfaces
  )
