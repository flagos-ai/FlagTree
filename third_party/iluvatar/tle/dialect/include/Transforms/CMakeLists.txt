set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -D__ILUVATAR_TLE__)
add_public_tablegen_target(IluvatarTleTransformsIncGen)
