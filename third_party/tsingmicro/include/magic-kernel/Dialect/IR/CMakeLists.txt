set(LLVM_TARGET_DEFINITIONS MagicKernelOps.td)
mlir_tablegen(MagicKernelDialect.h.inc -gen-dialect-decls -dialect=mk)
mlir_tablegen(MagicKernelDialect.cpp.inc -gen-dialect-defs -dialect=mk)
mlir_tablegen(MagicKernelOps.h.inc -gen-op-decls)
mlir_tablegen(MagicKernelOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS MagicKernelTypes.td)
mlir_tablegen(MagicKernelTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MagicKernelTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(MagicKernelTableGen)
