set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/TLEDialect.h
  ${CMAKE_CURRENT_BINARY_DIR}/TLEDialect.h
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/TLEOps.h
  ${CMAKE_CURRENT_BINARY_DIR}/TLEOps.h
  COPYONLY
)

set(LLVM_TARGET_DEFINITIONS TLEDialect.td)
mlir_tablegen(TLEDialect.h.inc -gen-dialect-decls -dialect=xtle)
mlir_tablegen(TLEDialect.cpp.inc -gen-dialect-defs -dialect=xtle)

set(LLVM_TARGET_DEFINITIONS TLEOps.td)
mlir_tablegen(TLEOps.h.inc -gen-op-decls)
mlir_tablegen(TLEOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(TLETableGen)
