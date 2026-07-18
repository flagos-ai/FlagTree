set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS GCUWSOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=gcuws)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=gcuws)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=gcuws)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=gcuws)

add_mlir_doc(GCUWSDialect GCUWSDialect dialects/ -gen-dialect-doc)
add_mlir_doc(GCUWSOps GCUWSOps dialects/ -gen-op-doc)
add_public_tablegen_target(GCUWSTableGen_${arch})

set(LLVM_TARGET_DEFINITIONS GCUWSAttrDefs.td)
mlir_tablegen(GCUWSAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(GCUWSAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(GCUWSAttrEnums.h.inc -gen-enum-decls)
mlir_tablegen(GCUWSAttrEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS GCUWSOpInterfaces.td)
mlir_tablegen(GCUWSOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(GCUWSOpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(GCUWSAttrDefsIncGen_${arch})
