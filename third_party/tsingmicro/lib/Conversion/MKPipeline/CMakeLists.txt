add_triton_library(MKPipeline
  MKPipelinePass.cpp
  MKLoopBoundCanonicalizePass.cpp
  DEPENDS
  MKPipelinePassIncGen
  MagicKernelTableGen
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonSharedUtils
)
