add_triton_library(TritonToAnnotation
  TritonToAnnotation.cpp

  DEPENDS
  TritonToAnnotationConversionPassIncGen

  LINK_LIBS
  BiShengIRAnnotationDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
