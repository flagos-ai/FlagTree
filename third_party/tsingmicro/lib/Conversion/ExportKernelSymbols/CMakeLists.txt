add_triton_library(ExportKernelSymbols
  ExportKernelSymbols.cpp

  DEPENDS
  ExportKernelSymbolsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
