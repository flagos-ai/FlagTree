set(LLVM_TARGET_DEFINITIONS TritonTypeInterfaces.td)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(TritonTypeInterfacesIncGen)
