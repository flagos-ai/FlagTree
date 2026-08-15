set(LLVM_TARGET_DEFINITIONS CGAEncodingAttr.td)
mlir_tablegen(CGAEncodingAttr.h.inc -gen-attrdef-decls)
add_public_tablegen_target(TritonGPUCGAAttrIncGen)

set(LLVM_TARGET_DEFINITIONS TritonGPUEnums.td)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TritonGPUOpsEnumsIncGen)
