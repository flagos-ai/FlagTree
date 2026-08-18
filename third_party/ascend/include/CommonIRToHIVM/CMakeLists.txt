set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name CommonIRToHIVM)
add_public_tablegen_target(CommonIRToHIVMConversionPassIncGen)
