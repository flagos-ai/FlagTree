add_triton_library(LinalgFusion
  LinalgFusion.cpp
  LinalgFusionPass.cpp

  DEPENDS
  LinalgFusionConversionPassIncGen

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
