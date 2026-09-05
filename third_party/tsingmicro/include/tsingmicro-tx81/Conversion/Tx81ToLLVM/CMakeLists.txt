set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name Tx81ToLLVM)
add_public_tablegen_target(Tx81ToLLVMConversionPassIncGen)

set(LLVM_TARGET_DEFINITIONS KernelArgBufferPass.td)
mlir_tablegen(KernelArgBufferPass.h.inc -gen-pass-decls --name KernelArgBufferPass)
add_public_tablegen_target(KernelArgBufferPassIncGen)
