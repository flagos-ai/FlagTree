set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS MathExtOps.td)
mlir_tablegen(MathExtOps.h.inc -gen-op-decls)
mlir_tablegen(MathExtOps.cpp.inc -gen-op-defs)
# mlir_tablegen(MathExtOpsEnums.h.inc -gen-enum-decls)
# mlir_tablegen(MathExtOpsEnums.cpp.inc -gen-enum-defs)
# mlir_tablegen(MathExtOpsAttributes.h.inc -gen-attrdef-decls)
# mlir_tablegen(MathExtOpsAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_doc(MathExtOps MathExtOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MathExtDialect.td)
mlir_tablegen(MathExtDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MathExtDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(MathExtDialect MathExtDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MathExtTypes.td)
mlir_tablegen(MathExtTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MathExtTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(MathExtTableGen)
