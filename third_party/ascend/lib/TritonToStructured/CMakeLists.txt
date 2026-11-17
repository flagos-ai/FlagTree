add_triton_library(TritonToStructured
        TritonToStructuredPass.cpp
        PtrAnalysis.cpp
        CannonicalizerConverter.cpp
        MemOpConverter.cpp
        MaskAnalysis.cpp

  DEPENDS
  TritonToStructuredConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
  BiShengIRHIVMDialect
)
