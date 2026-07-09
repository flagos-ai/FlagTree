set(LLVM_TARGET_DEFINITIONS TritonXPUOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
# add_mlir_doc(TritonXPUOps TritonXPUOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TritonXPUEnums.td)
mlir_tablegen(TritonXPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonXPUEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TritonXPUDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=triton_xpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=triton_xpu)
# add_mlir_doc(TritonXPUDialect TritonXPUDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TritonXPUTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=triton_xpu)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=triton_xpu)

set(LLVM_TARGET_DEFINITIONS TritonXPUInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(TritonXPUTableGen)


set(LLVM_TARGET_DEFINITIONS TritonXPUAttrDefs.td)
mlir_tablegen(TritonXPUAttrInterfaces.h.inc -gen-attr-interface-decls) # TritonXPU_AttrTrait
mlir_tablegen(TritonXPUAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(TritonXPUAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=triton_xpu)
mlir_tablegen(TritonXPUAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=triton_xpu)
add_public_tablegen_target(TritonXPUAttrDefsIncGen)
