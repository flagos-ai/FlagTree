add_triton_library(TritonToLinalg
        TritonToLinalgPass.cpp
        LoadStoreConverter.cpp
        TritonOpConverter.cpp
        BlockPtrAnalysis.cpp
        ImplicitPermute.cpp
        DescriptorConverter.cpp
        MarkTensorKindPass.cpp
        DeduplicateDebugNopsPass.cpp
        CanonicalizeDebugLocationsPass.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  TritonIR
  TritonTransforms
  TritonAnalysis
  TritonMemoryAccessAnalysis
  TritonToGraph
  TritonToLinalgGeneric
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
)
