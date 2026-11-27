add_mlir_dialect(AddressOps addr)

set(LLVM_TARGET_DEFINITIONS AddressBase.td)
mlir_tablegen(AddressOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AddressOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAddressOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS AddressOps.td)
mlir_tablegen(AddressOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AddressOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAddressOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS AddressOps.td)
mlir_tablegen(
  AddressOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=addr
)
mlir_tablegen(
  AddressOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=addr
)
add_public_tablegen_target(MLIRAddressOpsAttributesIncGen)
