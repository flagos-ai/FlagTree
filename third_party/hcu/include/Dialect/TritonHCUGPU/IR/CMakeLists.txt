set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonHCUGPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=hcug)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=hcug)
mlir_tablegen(OpsConversions.inc -gen-llvmir-conversions)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TritonHCUGPUDialect TritonHCUGPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TritonHCUGPUOps TritonHCUGPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(TritonHCUGPUTableGen)

set(LLVM_TARGET_DEFINITIONS TritonHCUGPUAttrDefs.td)
mlir_tablegen(TritonHCUGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonHCUGPUEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TritonHCUGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonHCUGPUAttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS TritonHCUGPUOpInterfaces.td)
mlir_tablegen(TritonHCUGPUOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TritonHCUGPUOpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TritonHCUGPUAttrDefsIncGen)
