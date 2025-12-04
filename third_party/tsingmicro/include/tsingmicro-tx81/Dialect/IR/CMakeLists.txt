set(LLVM_TARGET_DEFINITIONS Tx81Ops.td)
mlir_tablegen(Tx81Dialect.h.inc -gen-dialect-decls -dialect=tx)
mlir_tablegen(Tx81Dialect.cpp.inc -gen-dialect-defs -dialect=tx)
mlir_tablegen(Tx81Ops.h.inc -gen-op-decls)
mlir_tablegen(Tx81Ops.cpp.inc -gen-op-defs)

mlir_tablegen(Tx81Enums.h.inc -gen-enum-decls)
mlir_tablegen(Tx81Enums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Tx81Types.td)
mlir_tablegen(Tx81Types.h.inc -gen-typedef-decls)
mlir_tablegen(Tx81Types.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(Tx81TableGen)
