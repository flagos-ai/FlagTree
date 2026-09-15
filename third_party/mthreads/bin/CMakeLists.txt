get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(triton-opt triton-opt.cpp)

target_compile_options(triton-opt PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # FlagPrism: link the external Proton test-pass registration used by the
  # unified RegisterTritonDialects path.
  $<$<BOOL:${TRITON_BUILD_FLAGPRISM}>:TritonTestProton>
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)

add_executable(triton-reduce triton-reduce.cpp)
mlir_check_all_link_libraries(triton-reduce)
target_compile_options(triton-reduce PRIVATE -fno-rtti -fno-exceptions)

target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # FlagPrism: link the external Proton test-pass registration used by the
  # unified RegisterTritonDialects path.
  $<$<BOOL:${TRITON_BUILD_FLAGPRISM}>:TritonTestProton>
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-reduce)


add_executable(triton-llvm-opt triton-llvm-opt.cpp)
add_dependencies(triton-llvm-opt intrinsics_gen)
target_compile_options(triton-llvm-opt PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR
  TritonMUSALLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  )
export_executable_symbols_for_plugins(triton-llvm-opt)


add_executable(triton-tensor-layout triton-tensor-layout.cpp)
target_compile_options(triton-tensor-layout PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(triton-tensor-layout PRIVATE
  ${triton_libs}
  # FlagPrism: link the external Proton test-pass registration used by the
  # unified RegisterTritonDialects path.
  $<$<BOOL:${TRITON_BUILD_FLAGPRISM}>:TritonTestProton>
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  )
