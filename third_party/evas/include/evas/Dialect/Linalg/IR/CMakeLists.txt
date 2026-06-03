set(LLVM_TARGET_DEFINITIONS LinalgEnumsExt.td)
mlir_tablegen(LinalgOpsExtEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinalgOpsExtEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS LinalgOpsExt.td)
mlir_tablegen(LinalgOpsExt.h.inc -gen-op-decls)
mlir_tablegen(LinalgOpsExt.cpp.inc -gen-op-defs)
add_public_tablegen_target(EvasLinalgOpsExtIncGen)
