add_triton_library(TritonToHIVM
  TritonToHIVM.cpp

  DEPENDS
  TritonToHIVMConversionPassIncGen
  TleDSATableGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  TleDSAToHIVM
)
