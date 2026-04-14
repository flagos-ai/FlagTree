set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS MemrefExtDialect.td)
mlir_tablegen(MemrefExtDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MemrefExtDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS MemrefExtOps.td)
mlir_tablegen(MemrefExtOps.h.inc -gen-op-decls)
mlir_tablegen(MemrefExtOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MemrefExtTableGen${arch})
