add_triton_library(TLEToMagicKernel
  TLEToMK.cpp
  TLEToMKPass.cpp

  DEPENDS
  MagicKernelTableGen
  TLEToMKConversionPassIncGen
  TleDsaDialectIncGen
  TleDsaTypesIncGen
  TleDsaOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRArithDialect
  TritonIR
  TritonStructuredIR
  TleDsaIR
)
