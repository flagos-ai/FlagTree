set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS McTleOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(McTleOps McTleOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS McTleDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=mctle)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=mctle)
add_mlir_doc(McTleDialect McTleDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS McTleAttrDefs.td)
mlir_tablegen(AttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(AttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(McTleTableGen)
