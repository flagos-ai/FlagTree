get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-adapter-opt triton-adapter-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-adapter-opt)
target_link_libraries(triton-adapter-opt PRIVATE
  TritonToAnnotation
  TritonToHIVM
  DiscreteMaskAccessConversion
  TritonToLinalg
  TritonToUnstructure
  TritonTransforms
  ${dialect_libs}
  ${conversion_libs}
  TritonGPUTransforms
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-adapter-opt)