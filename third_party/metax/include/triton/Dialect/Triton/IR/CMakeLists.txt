set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(_flagtree_metax_triton_spec_dir
  "${PROJECT_SOURCE_DIR}/third_party/metax/spec_cpp/include/triton/Dialect/Triton/IR")

set(LLVM_TARGET_DEFINITIONS
  "${_flagtree_metax_triton_spec_dir}/TritonOps.td")
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
string(REGEX REPLACE "\\.td$" "" _flagtree_metax_triton_ops_doc_stem
  "${_flagtree_metax_triton_spec_dir}/TritonOps.td")
add_mlir_doc(
  "${_flagtree_metax_triton_ops_doc_stem}" TritonOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(TritonDialect TritonDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TritonTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TritonInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS TritonOpInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TritonTableGen)
