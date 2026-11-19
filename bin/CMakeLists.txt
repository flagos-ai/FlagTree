get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

find_package(Threads REQUIRED)

# The pub-LLVM22 frontend links gcc-11's static libstdc++.a (resolved by
# setup.py into _GCC11_LIBSTDCXX) AFTER the MLIR/LLVM archives, since those
# archives reference internal libstdc++ symbols that the host gcc-9
# libstdc++.a does not provide.  Applied per-target at the end of each
# target_link_libraries below.
function(triton_link_gcc11_libstdcxx target)
  if(_GCC11_LIBSTDCXX)
    target_link_libraries(${target} PRIVATE "${_GCC11_LIBSTDCXX}")
  endif()
endfunction()

add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-opt)
target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  Threads::Threads
)

mlir_check_all_link_libraries(triton-opt)
triton_link_gcc11_libstdcxx(triton-opt)

add_llvm_executable(triton-reduce triton-reduce.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(triton-reduce)

llvm_update_compile_flags(triton-reduce)
target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  Threads::Threads
)

mlir_check_all_link_libraries(triton-reduce)
triton_link_gcc11_libstdcxx(triton-reduce)

add_llvm_executable(triton-lsp triton-lsp.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-lsp)
target_link_libraries(triton-lsp PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRLspServerLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  Threads::Threads
)

mlir_check_all_link_libraries(triton-lsp)
triton_link_gcc11_libstdcxx(triton-lsp)


add_llvm_executable(triton-llvm-opt
  triton-llvm-opt.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen

  Threads::Threads
  )
if(BUILD_MCTLE)
  add_dependencies(triton-llvm-opt McTleIR)
  target_sources(triton-llvm-opt PRIVATE
    $<TARGET_OBJECTS:McTleIR>
  )
endif()
if(_GCC11_LIBSTDCXX)
  target_link_libraries(triton-llvm-opt PRIVATE "${_GCC11_LIBSTDCXX}")
endif()
export_executable_symbols_for_plugins(triton-llvm-opt)


add_llvm_executable(triton-tensor-layout triton-tensor-layout.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(triton-tensor-layout PRIVATE
  ${triton_libs}
  TritonTestAnalysis
  TritonTestDialect
  TritonTestProton
  TritonAMDGPUTestAnalysis
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  Threads::Threads
  )
triton_link_gcc11_libstdcxx(triton-tensor-layout)
