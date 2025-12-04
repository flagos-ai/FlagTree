set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name Tx81MemrefToLLVM)
add_public_tablegen_target(Tx81MemrefToLLVMConversionPassIncGen)
