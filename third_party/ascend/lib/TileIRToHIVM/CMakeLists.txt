add_triton_library(TileIRToHIVM
  TileIRToHIVM.cpp

  DEPENDS
  TileIRToHIVMConversionPassIncGen

  LINK_LIBS
  TileIRIR
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)