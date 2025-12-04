add_triton_library(MagicKernelIR
  IR/MagicKernelDialect.cpp
  Transforms/BufferizableOpInterfaceImpl.cpp

  DEPENDS
  MagicKernelTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
