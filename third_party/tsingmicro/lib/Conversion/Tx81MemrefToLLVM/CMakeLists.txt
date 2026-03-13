add_triton_library(Tx81MemrefToLLVM
  Tx81MemrefToLLVM.cpp
  Tx81MemrefToLLVMPass.cpp

  DEPENDS
  Tx81MemrefToLLVMConversionPassIncGen
  TritonSharedUtils

  LINK_LIBS PUBLIC
  TritonSharedUtils
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
