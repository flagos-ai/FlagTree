set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS IluvatarTleOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=iluvatar_tle -D__ILUVATAR_TLE__)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=iluvatar_tle -D__ILUVATAR_TLE__)
mlir_tablegen(Ops.h.inc -gen-op-decls -D__ILUVATAR_TLE__)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -D__ILUVATAR_TLE__)

if(FLAGCX_ENABLED)
set(LLVM_TARGET_DEFINITIONS IluvatarFlagCxOps.td)
mlir_tablegen(FlagCxOps.h.inc -gen-op-decls -D__ILUVATAR_TLE__)
mlir_tablegen(FlagCxOps.cpp.inc -gen-op-defs -D__ILUVATAR_TLE__)
endif()

set(LLVM_TARGET_DEFINITIONS IluvatarTleAttrDefs.td)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls -D__ILUVATAR_TLE__)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs -D__ILUVATAR_TLE__)
mlir_tablegen(IluvatarTleAttrDefs.h.inc -gen-attrdef-decls -D__ILUVATAR_TLE__)
mlir_tablegen(IluvatarTleAttrDefs.cpp.inc -gen-attrdef-defs -D__ILUVATAR_TLE__)

add_public_tablegen_target(IluvatarTleTableGen)
