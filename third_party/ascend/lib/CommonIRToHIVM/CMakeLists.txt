add_triton_library(CommonIRToHIVM
  CommonIRToHIVM.cpp
  ReduceSumStrengthPass.cpp
  TensorViewAccessAnalysis.cpp
  TensorViewStrategy.cpp
  TensorViewLowering.cpp

  DEPENDS
  CommonIRToHIVMConversionPassIncGen

  LINK_LIBS
  CommonIRIR
  TensorViewIR
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRArithDialect
)
