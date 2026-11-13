# Added after third_party/ascend/lib, so TRITON_LIBS already holds the ascend
# conversion libraries; the tools below do not have to list them one by one.
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# triton-opt: replays the ttir->ttadapter pipeline that backend/compiler.py
# prints under `if opt.debug:`. Its registry is self-contained (see the comment
# at the top of triton-opt.cpp for why core bin/RegisterTritonDialects.h is not
# reused).
add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-opt)

target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTransforms
  # The pipeline runs `inline` on func.func right after commonir-to-hivm.
  MLIRFuncInlinerExtension
  # Core BishengIR dialect libraries
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
)

mlir_check_all_link_libraries(triton-opt)

# triton-mlir-opt: Triton-Ascend optimizer combining MLIR and BishengIR dialects
add_llvm_executable(triton-mlir-opt triton-mlir-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-mlir-opt)

target_link_libraries(triton-mlir-opt PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  # Core BishengIR dialect libraries
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
)

mlir_check_all_link_libraries(triton-mlir-opt)


# Emit into the build tree, NOT into CMAKE_LIBRARY_OUTPUT_DIRECTORY (the wheel's
# triton/_C). python/setup_tools/utils/ascend.py install_extension copies these
# into triton/_C and strips the copies, which keeps a full-symbol original here
# for source-level debugging. Writing straight to triton/_C would leave only one
# file: stripping it would discard the symbols for good, and because strip does
# not make the target look out of date, an incremental build would not restore
# them either.
set_target_properties(triton-opt triton-mlir-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/third_party/ascend/bin"
)
