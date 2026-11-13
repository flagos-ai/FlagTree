add_triton_library(TritonToUnstructure
        IndirectAtomicUtils.cpp
        UnstructureConversionPass.cpp
        OffsetAnalysis.cpp
        BubbleUpOperation.cpp
        ReplaceArguments.cpp

  DEPENDS
  TritonToUnstructureConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonAnalysis
  TritonMemoryAccessAnalysis
  MLIRSCFTransforms
)
