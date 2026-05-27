add_triton_library(TritonToHIVM
  TritonToHIVM.cpp

  DEPENDS
    TritonToHIVMConversionPassIncGen
    TritonCanonicalizeIncGen
    TritonAscendTableGen
    TritonAscendAttrDefsIncGen

  LINK_LIBS
    BiShengIRHIVMDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    TritonIR
)
