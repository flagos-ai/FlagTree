set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name DiscreteMaskAccessConversion)
add_public_tablegen_target(DiscreteMaskAccessConversionPassIncGen)