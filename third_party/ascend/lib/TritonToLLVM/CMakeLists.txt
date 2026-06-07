add_triton_library(TritonToLLVM
  TritonToLLVM.cpp

  DEPENDS
  TritonToLLVMConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
