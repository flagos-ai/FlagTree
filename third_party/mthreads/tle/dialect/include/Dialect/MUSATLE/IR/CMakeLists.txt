set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS MUSATLEOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=musa_tle -D__TLE__)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=musa_tle -D__TLE__)
mlir_tablegen(Ops.h.inc -gen-op-decls -D__TLE__)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -D__TLE__)
add_public_tablegen_target(MUSATLETableGen)
