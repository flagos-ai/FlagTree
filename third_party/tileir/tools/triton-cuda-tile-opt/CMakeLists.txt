get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-cuda-tile-opt triton-cuda-tile-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-cuda-tile-opt)
target_link_libraries(triton-cuda-tile-opt PRIVATE
  TritonTransforms
  TritonTileIRTransforms
  TritonToTileIR
  TritonNvidiaGPUTransforms
  # tests
  TritonTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-cuda-tile-opt)
