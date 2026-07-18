set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonILUVATARGPU)
add_public_tablegen_target(TritonILUVATARGPUTransformsIncGen)
