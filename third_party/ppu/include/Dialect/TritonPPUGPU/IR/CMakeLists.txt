set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonPPUGPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ttpg)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=ttpg)
mlir_tablegen(OpsConversions.inc -gen-llvmir-conversions)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TritonPPUGPUDialect TritonPPUGPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TritonPPUGPUOps TritonPPUGPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(TritonPPUGPUTableGen)

set(LLVM_TARGET_DEFINITIONS TritonPPUGPUAttrDefs.td)
mlir_tablegen(TritonPPUGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonPPUGPUAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TritonPPUGPUAttrDefsIncGen)
