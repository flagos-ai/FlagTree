get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(spine-triton-opt spine-triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(spine-triton-opt)
target_link_libraries(spine-triton-opt PRIVATE
  TritonTransforms
  TritonSharedAnalysis
  ProtonIR
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)

mlir_check_all_link_libraries(spine-triton-opt)
