set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLVMXPU)
add_public_tablegen_target(LLVMXPUTransformsIncGen)
