get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(evas-triton-opt evas-triton-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(evas-triton-opt)
target_include_directories(evas-triton-opt PRIVATE
  ${TRITON_SHARED_TOOLS_DIR}
)
target_link_libraries(evas-triton-opt PRIVATE
  EvasTritonToEvas
  EVLinalgTransforms
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIRMemRefTransforms
  MLIRSparseTensorTransforms
  MLIRControlFlowTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTransforms
)

mlir_check_all_link_libraries(evas-triton-opt)
