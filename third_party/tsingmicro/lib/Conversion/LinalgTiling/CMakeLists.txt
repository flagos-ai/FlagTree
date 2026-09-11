add_triton_library(LinalgTiling
  LinalgTiling.cpp
  LinalgTilingPass.cpp

  DEPENDS
  LinalgTilingConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
