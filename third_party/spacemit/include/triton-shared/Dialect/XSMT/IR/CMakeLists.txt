set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XSMTDialect.h
  ${CMAKE_CURRENT_BINARY_DIR}/XSMTDialect.h
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XSMTAttrs.h
  ${CMAKE_CURRENT_BINARY_DIR}/XSMTAttrs.h
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XSMTOps.h
  ${CMAKE_CURRENT_BINARY_DIR}/XSMTOps.h
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XSMTTypes.h
  ${CMAKE_CURRENT_BINARY_DIR}/XSMTTypes.h
  COPYONLY
)

set(LLVM_TARGET_DEFINITIONS XSMTDialect.td)
mlir_tablegen(XSMTDialect.h.inc -gen-dialect-decls -dialect=xsmt)
mlir_tablegen(XSMTDialect.cpp.inc -gen-dialect-defs -dialect=xsmt)

set(LLVM_TARGET_DEFINITIONS XSMTOps.td)
mlir_tablegen(XSMTOps.h.inc -gen-op-decls)
mlir_tablegen(XSMTOps.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS XSMTTypes.td)
mlir_tablegen(XSMTTypes.h.inc -gen-typedef-decls)
mlir_tablegen(XSMTTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS XSMTAttrs.td)
mlir_tablegen(XSMTAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(XSMTAttrs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(XSMTTableGen)
