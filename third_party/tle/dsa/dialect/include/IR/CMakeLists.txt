# Copyright 2026- Xcoresigma Technology Co., Ltd

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TleDSAOps.td)
mlir_tablegen(TleDSAOps.h.inc -gen-op-decls)
mlir_tablegen(TleDSAOps.cpp.inc -gen-op-defs)
add_mlir_doc(TleDSAOps TleDSAOps dialects/ -gen-op-doc)

# DSA ops are registered into the main tle dialect. Do not generate a second
# dialect class for the same dialect namespace here.
#set(LLVM_TARGET_DEFINITIONS TleDSADialect.td)
#mlir_tablegen(TleDSADialect.h.inc -gen-dialect-decls -dialect=tle)
#mlir_tablegen(TleDSADialect.cpp.inc -gen-dialect-defs -dialect=tle)
#add_mlir_doc(TleDSADialect TleDSADialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TleDSAAttrDefs.td)
mlir_tablegen(TleDSAAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TleDSAAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TleDSATableGen)
