get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# TODO: Workaround include path
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tsingmicro/include)
# include_directories(${CMAKE_CURRENT_BINARY_DIR}/../third_party/tsingmicro/include)

add_llvm_executable(tsingmicro-opt tsingmicro-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(tsingmicro-opt)
target_link_libraries(tsingmicro-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${triton_libs}
  ZTCAnalysis
  # tests
  TritonTestAnalysis
  TritonTestDialectTritonGPU
  TritonAMDGPUTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRMathTestPasses
)

mlir_check_all_link_libraries(tsingmicro-opt)

# add_llvm_executable(tsingmicro-reduce tsingmicro-reduce.cpp PARTIAL_SOURCES_INTENDED)
# mlir_check_all_link_libraries(tsingmicro-reduce)

# llvm_update_compile_flags(tsingmicro-reduce)
# target_link_libraries(tsingmicro-reduce PRIVATE
#   ${dialect_libs}
#   ${conversion_libs}
#   ${extension_libs}
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialectTritonGPU
#   TritonAMDGPUTestAnalysis
#   # MLIR core
#   MLIRReduceLib
#   MLIRPass
#   MLIRTransforms
#   MLIRMathTestPasses
# )

# mlir_check_all_link_libraries(tsingmicro-reduce)

# add_llvm_executable(tsingmicro-lsp tsingmicro-lsp.cpp PARTIAL_SOURCES_INTENDED)

# llvm_update_compile_flags(tsingmicro-lsp)
# target_link_libraries(tsingmicro-lsp PRIVATE
#   ${dialect_libs}
#   ${conversion_libs}
#   ${extension_libs}
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialectTritonGPU
#   TritonAMDGPUTestAnalysis
#   # MLIR core
#   MLIRLspServerLib
#   MLIRPass
#   MLIRTransforms
#   MLIRMathTestPasses
# )

# mlir_check_all_link_libraries(tsingmicro-lsp)


# add_llvm_executable(tsingmicro-llvm-opt
#   tsingmicro-llvm-opt.cpp

#   PARTIAL_SOURCES_INTENDED
#   DEPENDS
#   intrinsics_gen
#   SUPPORT_PLUGINS
#   )
# target_link_libraries(tsingmicro-llvm-opt PRIVATE
#   TritonLLVMIR

#   LLVMAnalysis
#   LLVMCore
#   LLVMSupport
#   LLVMOption
#   LLVMCodeGen
#   )
# export_executable_symbols_for_plugins(tsingmicro-llvm-opt)


# add_llvm_executable(tsingmicro-tensor-layout tsingmicro-tensor-layout.cpp PARTIAL_SOURCES_INTENDED)
# target_link_libraries(tsingmicro-tensor-layout PRIVATE
#   ${triton_libs}
#   ${conversion_libs}
#   ${extension_libs}
#   ${dialect_libs}
#   TritonTestAnalysis
#   TritonTestDialectTritonGPU
#   TritonAMDGPUTestAnalysis
#   MLIRMathTestPasses
#   )

install(TARGETS tsingmicro-opt
  RUNTIME DESTINATION ${INSTALL_TSINGMICRO_DIR}/bin
)
