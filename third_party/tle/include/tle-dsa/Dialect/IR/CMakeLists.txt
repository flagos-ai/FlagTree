set(LLVM_TARGET_DEFINITIONS DsaDialect.td)
mlir_tablegen(DsaOpsDialect.h.inc -gen-dialect-decls -dialect=dsa)
mlir_tablegen(DsaOpsDialect.cpp.inc -gen-dialect-defs -dialect=dsa)
add_public_tablegen_target(TleDsaDialectIncGen)

set(LLVM_TARGET_DEFINITIONS DsaDialect.td)
mlir_tablegen(DsaOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=dsa)
mlir_tablegen(DsaOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=dsa)
add_public_tablegen_target(TleDsaTypesIncGen)

set(LLVM_TARGET_DEFINITIONS DsaOps.td)
mlir_tablegen(DsaOps.h.inc -gen-op-decls)
mlir_tablegen(DsaOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TleDsaOpsIncGen)
