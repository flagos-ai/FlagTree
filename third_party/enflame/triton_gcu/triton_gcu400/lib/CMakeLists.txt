# Build subdirectories first (they define the libraries that core depends on)
add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)

# -----------------------------------------------------------------------------
# Core Library (libtriton_gcu400_core.so)
# -----------------------------------------------------------------------------
# GCU-specific libraries + Triton MLIR targets (all from tritonlang_llvm_build).
set(_core_dialect_libs
  GCUIR_gcu400
  TritonGCUIR_gcu400
  MLIRTritonToGCU_gcu400
  MLIRGCUTritonToTritonGPU_gcu400
  MLIRTritonGCUTransforms_gcu400
  MemrefExtIR_gcu400
  MathExtIR_gcu400
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRAffineDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
  MLIROptLib
  MLIRAnalysis
  MLIRIR
  MLIRSupport
  GCUWSIR_gcu400
)

if(FLAGTREE_TLE)
  list(APPEND _core_dialect_libs
    TleIR
    TleToLLVM
    TritonTLETransforms
    TritonTLEAnalysis
    TritonNVIDIAGPUToLLVM
    NVGPUIR
  )
endif()

add_library(triton_gcu400_core SHARED triton_gcu400_core.cpp ${TRITON_OUTPUT_FILES})
target_compile_features(triton_gcu400_core PRIVATE cxx_std_17)
set_source_files_properties(${TRITON_OUTPUT_FILES} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)

target_link_libraries(triton_gcu400_core PRIVATE
  ${_core_dialect_libs}
  ${TRITON_MLIR_DEPS}
)
mlir_check_all_link_libraries(triton_gcu400_core)
target_link_options(triton_gcu400_core PRIVATE -Wl,--no-undefined)

target_include_directories(triton_gcu400_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../include
  ${CMAKE_SOURCE_DIR}/gcu_compiler/include
  ${CMAKE_SOURCE_DIR}/gcu_compiler/lib
  ${CMAKE_BINARY_DIR}/gcu_compiler/include
)

target_include_directories(triton_gcu400_core SYSTEM PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${TRITON_SOURCE_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_SOURCE_DIR}
  ${TRITON_BINARY_DIR}/lib/Dialect/Triton/Transforms
  ${TRITON_SOURCE_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party
)

add_dependencies(triton_gcu400_core triton_gcu400)
if(TARGET MathExtTableGen_gcu400)
  add_dependencies(triton_gcu400_core MathExtTableGen_gcu400 MemrefExtTableGen_gcu400)
endif()

set_target_properties(triton_gcu400_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu400_core.map"
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu400_core.map
)

message(STATUS "[gcu400] Built core library: libtriton_gcu400_core.so")

# Install core library
install(TARGETS triton_gcu400_core
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)
