add_triton_library(CommonIRToHIVM
  CommonIRToHIVM.cpp

  DEPENDS
  CommonIRToHIVMConversionPassIncGen

  LINK_LIBS
  CommonIRIR
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
)
