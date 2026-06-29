add_triton_library(EraseLinalgCasts
  EraseLinalgCasts.cpp

  DEPENDS
  EraseLinalgCastsConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIRBufferizationDialect
  TritonIR
)
