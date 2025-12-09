set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS FlagTreeOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(FlagTreeOps FlagTreeOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FlagTreeDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=flagtree)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=flagtree)
add_mlir_doc(FlagTreeDialect FlagTreeDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS FlagTreeAttrDefs.td)
mlir_tablegen(FlagTreeAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(FlagTreeAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(FlagTreeTableGen)
