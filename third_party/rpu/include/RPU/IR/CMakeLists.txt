set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS RPUDialect.td)
mlir_tablegen(RPUDialect.h.inc -gen-dialect-decls -dialect=rpu)
mlir_tablegen(RPUDialect.cpp.inc -gen-dialect-defs -dialect=rpu)

set(LLVM_TARGET_DEFINITIONS RPUTypes.td)
mlir_tablegen(RPUTypes.h.inc -gen-typedef-decls -typedefs-dialect=rpu)
mlir_tablegen(RPUTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=rpu)

set(LLVM_TARGET_DEFINITIONS RPUOps.td)
mlir_tablegen(RPUOps.h.inc -gen-op-decls)
mlir_tablegen(RPUOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(RPUTableGen)
