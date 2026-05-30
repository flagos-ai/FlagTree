set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS RPUPlanOps.td)
mlir_tablegen(RPUPlanDialect.h.inc -gen-dialect-decls -dialect=rpu_plan)
mlir_tablegen(RPUPlanDialect.cpp.inc -gen-dialect-defs -dialect=rpu_plan)
mlir_tablegen(RPUPlanOps.h.inc -gen-op-decls)
mlir_tablegen(RPUPlanOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RPUPlanTableGen)
