# Build subdirectories first (they define the libraries that core depends on)
add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)

# -----------------------------------------------------------------------------
# Core Library (libtriton_gcu300_core.so)
# -----------------------------------------------------------------------------
# Use Triton's MLIR/LLVM libraries (from tritonlang_llvm_build)
# NOT gcu_compiler's LLVM to avoid conflicts.
# The Triton .o files embedded in this shared library reference many MLIR
# symbols (registrations, translations, conversion patterns), so we must
# link all the MLIR static libraries they depend on.
set(_core_dialect_libs
  GCUIR_gcu300
  TritonGCUIR_gcu300
  MLIRTritonToGCU_gcu300
  MLIRTritonGCUTransforms_gcu300
  MLIRGCUTritonToTritonGPU_gcu300
  MemrefExtIR_gcu300
  MathExtIR_gcu300
)

add_library(triton_gcu300_core SHARED triton_gcu300_core.cpp ${TRITON_OUTPUT_FILES})
target_compile_features(triton_gcu300_core PRIVATE cxx_std_17)
set_source_files_properties(${TRITON_OUTPUT_FILES} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)

target_link_libraries(triton_gcu300_core PRIVATE
  ${_core_dialect_libs}
  ${TRITON_MLIR_DEPS}
)
mlir_check_all_link_libraries(triton_gcu300_core)
target_link_options(triton_gcu300_core PRIVATE -Wl,--no-undefined)

target_include_directories(triton_gcu300_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../include
  ${CMAKE_SOURCE_DIR}/gcu_compiler/include
  ${CMAKE_SOURCE_DIR}/gcu_compiler/lib
  ${CMAKE_BINARY_DIR}/gcu_compiler/include
)

target_include_directories(triton_gcu300_core SYSTEM PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${TRITON_SOURCE_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_SOURCE_DIR}
  ${TRITON_BINARY_DIR}/lib/Dialect/Triton/Transforms
  ${TRITON_SOURCE_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party
)

add_dependencies(triton_gcu300_core triton_gcu300)
if(TARGET MathExtTableGen_gcu300)
  add_dependencies(triton_gcu300_core MathExtTableGen_gcu300 MemrefExtTableGen_gcu300)
endif()

set_target_properties(triton_gcu300_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu300_core.map"
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu300_core.map
)

message(STATUS "[gcu300] Built core library: libtriton_gcu300_core.so")

# Install core library
install(TARGETS triton_gcu300_core
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)
