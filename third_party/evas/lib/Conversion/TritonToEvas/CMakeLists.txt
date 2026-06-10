add_triton_library(EvasTritonToEvas
  TritonArithToLinalgNamed.cpp
  TritonToEvasPipeline.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizationTransforms
  MLIRPass
  MLIRTransforms
  StructuredToMemref
  TritonArithToLinalg
  TritonPtrToMemref
  TritonToLinalgExperimental
  TritonToStructured
  TritonToUnstructured
  UnstructuredToMemref
  EvasLinalgIR
  EVLinalgTransforms
)

target_compile_options(EvasTritonToEvas PRIVATE -Wno-deprecated-declarations)
