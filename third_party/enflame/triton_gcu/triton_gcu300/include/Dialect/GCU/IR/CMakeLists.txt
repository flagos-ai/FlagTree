set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS GCUOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OpsAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_doc(GCUOps GCUOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GCUDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(GCUDialect GCUDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS GCUTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS GCUInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

add_public_tablegen_target(GCUTableGen${arch})
