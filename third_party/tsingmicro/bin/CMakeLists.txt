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
  TritonSharedAnalysis
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonTestProton
  TritonAMDGPUTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIRTransforms
  MLIRMathTestPasses
  # Proton libraries (added after tsingmicro in top-level CMakeLists,
  # so not captured in ${triton_libs} at configure time)
  ProtonAnalysis
  ProtonToProtonGPU
  ProtonGPUToLLVM
  ProtonAMDGPUToLLVM
  ProtonNVIDIAGPUToLLVM
  ProtonGPUIR
  ProtonIR
  ProtonGPUTransforms
  # TLE DSA libraries (same ordering issue)
  TleDsaToCore
  TleDsaIR
)

mlir_check_all_link_libraries(tsingmicro-opt)

install(TARGETS tsingmicro-opt
  RUNTIME DESTINATION ${INSTALL_TSINGMICRO_DIR}/bin
)

if(TSINGMICRO_SYNC_TO_SOURCE)
  add_custom_command(TARGET tsingmicro-opt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TSINGMICRO_BACKEND_SRC_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tsingmicro-opt>
            ${TSINGMICRO_BACKEND_SRC_DIR}/bin/
    VERBATIM
  )
endif()
