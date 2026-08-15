set(LLVM_TARGET_DEFINITIONS TritonNvidiaGPUTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(TritonNvidiaGPUTypesIncGen)
