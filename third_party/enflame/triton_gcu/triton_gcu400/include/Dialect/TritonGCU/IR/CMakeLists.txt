set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonGCUDialect.td)
mlir_tablegen(TritonGCUDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TritonGCUDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TritonGCUTypes.td)
mlir_tablegen(TritonGCUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TritonGCUTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TritonGCUOps.td)
mlir_tablegen(TritonGCUOps.h.inc -gen-op-decls)
mlir_tablegen(TritonGCUOps.cpp.inc -gen-op-defs)
mlir_tablegen(TritonGCUOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonGCUOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TritonGCUOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonGCUOpsAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS TritonGCUInterfaces.td)
mlir_tablegen(TritonGCUAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TritonGCUAttrInterfaces.cpp.inc -gen-attr-interface-defs)

add_public_tablegen_target(TritonGCUTableGen_${arch})
