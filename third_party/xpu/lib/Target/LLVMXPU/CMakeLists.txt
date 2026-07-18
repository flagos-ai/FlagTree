add_triton_library(MLIRLLVMXPUToLLVMIRTranslation
  LLVMXPUToLLVMIRTranslation.cpp

  # DEPENDS
  # MLIRXPUConversionsIncGen

#   LINK_COMPONENTS
#   Core

#   LINK_LIBS PUBLIC
#   MLIRLLVMXPUDialect
#   MLIRIR
#   MLIRLLVMDialect
#   MLIRSupport
#   MLIRTargetLLVMIRExport
)

add_xpu_sdnn_object(MLIRLLVMXPUToLLVMIRTranslationSDNN
  ${CMAKE_CURRENT_SOURCE_DIR}/XPUToIRTranslationForSDNN.cpp.o
)
