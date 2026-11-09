add_triton_library(TritonToHIVM
  TritonToHIVM.cpp

  DEPENDS
  TritonToHIVMConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
