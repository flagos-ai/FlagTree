# flagtree tle
# IR headers for TLE dialect
# Add IR header files here

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TleOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TleOps TleOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TleDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tle)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tle)
add_mlir_doc(TleDialect TleDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TleAttrDefs.td)
mlir_tablegen(TleAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TleAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TleTableGen)
