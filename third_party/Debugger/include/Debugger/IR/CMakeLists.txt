set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS FlagTreeDebugOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(FlagTreeDebugOps FlagTreeDebugOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FlagTreeDebugDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=flagtree_debug)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=flagtree_debug)

add_public_tablegen_target(FlagTreeDebugTableGen)
