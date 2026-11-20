set(LLVM_TARGET_DEFINITIONS TritonStructuredDialect.td)
mlir_tablegen(TritonStructuredDialect.h.inc -gen-dialect-decls -dialect=tts)
mlir_tablegen(TritonStructuredDialect.cpp.inc -gen-dialect-defs -dialect=tts)
mlir_tablegen(TritonStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(TritonStructuredOps.cpp.inc -gen-op-defs)


add_public_tablegen_target(TritonStructuredTableGen)
