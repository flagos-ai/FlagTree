set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

file(RELATIVE_PATH patch_rel_dir "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "triton_patch" "third_party/triton" triton_rel_dir "${patch_rel_dir}")
set(triton_abs_dir "${CMAKE_SOURCE_DIR}/${triton_rel_dir}")
# message(STATUS "triton_abs_dir: ${triton_abs_dir}")

set(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
# add_mlir_doc(TritonOps TritonOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ${triton_abs_dir}/TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
# add_mlir_doc(TritonDialect TritonDialect dialects/ -gen-dialect-doc)

# TODO: When upgrading to Triton 3.4.0, enable the commented line below and remove the current line.
# set(LLVM_TARGET_DEFINITIONS ${triton_abs_dir}/TritonTypes.td)
set(LLVM_TARGET_DEFINITIONS TritonTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS ${triton_abs_dir}/TritonInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS ${triton_abs_dir}/TritonTypeInterfaces.td)
mlir_tablegen(TritonTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TritonTypeInterfaces.cpp.inc -gen-type-interface-defs)

# TODO: When upgrading to Triton 3.4.0, enable the commented line below and remove the current line.
# set(LLVM_TARGET_DEFINITIONS ${triton_abs_dir}/TritonOpInterfaces.td)
set(LLVM_TARGET_DEFINITIONS TritonOpInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(Patched_TritonTableGen)
