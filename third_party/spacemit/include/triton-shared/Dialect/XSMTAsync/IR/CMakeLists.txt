set(LLVM_TARGET_DEFINITIONS XSMTAsyncDialect.td)

mlir_tablegen(XSMTAsyncDialect.h.inc -gen-dialect-decls -dialect=xsmt_async)
mlir_tablegen(XSMTAsyncDialect.cpp.inc -gen-dialect-defs -dialect=xsmt_async)

set(LLVM_TARGET_DEFINITIONS XSMTAsyncOps.td)

mlir_tablegen(XSMTAsync.h.inc -gen-op-decls)

mlir_tablegen(XSMTAsync.cpp.inc -gen-op-defs)

add_public_tablegen_target(XSMTAsyncTableGen)
