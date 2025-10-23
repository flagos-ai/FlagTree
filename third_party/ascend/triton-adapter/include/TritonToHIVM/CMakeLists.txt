set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToHIVM)
add_public_tablegen_target(TritonToHIVMConversionPassIncGen)