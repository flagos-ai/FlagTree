add_triton_library(Tx81MemrefToLLVM
  Tx81MemrefToLLVM.cpp
  Tx81MemrefToLLVMPass.cpp

  DEPENDS
  Tx81MemrefToLLVMConversionPassIncGen
  TritonUtils

  LINK_LIBS PUBLIC
  TritonUtils
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
