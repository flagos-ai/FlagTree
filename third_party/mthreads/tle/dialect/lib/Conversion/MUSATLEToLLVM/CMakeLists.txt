add_triton_library(MUSATLEToLLVM
  LocalPointersOpToLLVM.cpp

  DEPENDS
  MUSATLETableGen

  LINK_LIBS PUBLIC
  MUSATLEIR
  TritonGPUToLLVM
  TritonGPUIR
)
