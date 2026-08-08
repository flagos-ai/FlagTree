# =============================================================================
# Triton GCU400 Build Configuration
# =============================================================================
#
# This file controls the build flow for GCU400 architecture by calling
# functions defined in cmake/triton_gcu.cmake.
#
# Build steps:
#   1. Reuse the in-tree Triton targets
#   2. Build core library (libtriton_gcu400_core.so)
#   3. Build tools (triton-gcu400-opt)
#   4. Build Python bindings (_triton_gcu400.so)
#   5. Setup unit tests
#
# =============================================================================

set(arch "gcu400")

if(FLAGTREE_TLE)
  add_definitions(-DENABLE_TLE)
endif()

# Load function library
include(triton_gcu)

# -----------------------------------------------------------------------------
# Step 1: Use the top-level in-tree Triton build
# -----------------------------------------------------------------------------
setup_triton_in_tree()

# Compiler flags for GCU400
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter")
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)

# Add Triton include directories for tablegen (needs to find .td files)
include_directories(SYSTEM ${TRITON_SOURCE_DIR}/include)
include_directories(SYSTEM ${TRITON_BINARY_DIR}/include)
include_directories(SYSTEM ${TRITON_SOURCE_DIR}/third_party)
include_directories(SYSTEM ${TRITON_BINARY_DIR}/third_party)

set(mlir_register_libs MLIRRegisterAllDialects MLIRRegisterAllExtensions MLIRRegisterAllPasses)

set(TRITON_MLIR_DEPS
  ${mlir_register_libs}
  MLIROptLib
  MLIRTranslateLib
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRToLLVMIRTranslationRegistration
  MLIRSPIRVTranslateRegistration
  MLIRTargetIRDLToCpp
  MLIRTargetWasmImport
  MLIRTargetCpp
  MLIRExportSMTLIB
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRControlFlowToLLVM
  MLIRMathToLLVM
  MLIRAMDGPUDialect
  MLIRGPUToROCDLTransforms
  MLIRGPUToNVVMTransforms
  MLIRUBToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFTransforms
  ${TRITON_CORE_LIBS}
)

add_library(triton_gcu400 INTERFACE)
target_include_directories(triton_gcu400 SYSTEM INTERFACE ${TRITON_SOURCE_DIR}/include)
target_include_directories(triton_gcu400 SYSTEM INTERFACE ${TRITON_BINARY_DIR}/include)
target_include_directories(triton_gcu400 SYSTEM INTERFACE ${TRITON_SOURCE_DIR})
target_include_directories(triton_gcu400 SYSTEM INTERFACE ${TRITON_BINARY_DIR}/lib/Dialect/Triton/Transforms)
target_include_directories(triton_gcu400 SYSTEM INTERFACE ${TRITON_SOURCE_DIR}/third_party)
target_include_directories(triton_gcu400 SYSTEM INTERFACE ${TRITON_BINARY_DIR}/third_party)

# -----------------------------------------------------------------------------
# Step 2: Build Dialects and Libraries
# -----------------------------------------------------------------------------
add_subdirectory(include)
add_subdirectory(lib)

# -----------------------------------------------------------------------------
# Step 3: Build Tools (triton-gcu400-opt)
# -----------------------------------------------------------------------------
# Standalone tool using only the C API from triton_gcu400_core
add_executable(triton-gcu400-opt triton-gcu400-opt.cpp)

set_target_properties(triton-gcu400-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH "$ORIGIN/../lib;$ORIGIN"
  INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN"
)

target_link_libraries(triton-gcu400-opt PRIVATE triton_gcu400_core)

# Install tools (omit from gcu300-only deb; still built for the wheel).
if(NOT KURAMA_GCU300_ONLY)
install(TARGETS triton-gcu400-opt RUNTIME DESTINATION bin)
endif()

add_custom_target(triton-gcu400-tools ALL DEPENDS triton-gcu400-opt)

# -----------------------------------------------------------------------------
# Step 4: Build Python Bindings
# -----------------------------------------------------------------------------
add_subdirectory(python)

# -----------------------------------------------------------------------------
# Step 5: Unit Tests
# -----------------------------------------------------------------------------
set(triton_gcu_source_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(_unittests_dir ${CMAKE_CURRENT_SOURCE_DIR}/../test/triton36/unittests)
if(EXISTS ${_unittests_dir})
  add_subdirectory(${_unittests_dir} unittests)
endif()

# -----------------------------------------------------------------------------
# Export Version Information
# -----------------------------------------------------------------------------
set_property(GLOBAL PROPERTY TRITON_GCU400_VERSION ${TRITON_ORIG_VERSION})

message(STATUS "[gcu400] Build configuration complete")
message(STATUS "[gcu400] Triton version: ${TRITON_ORIG_VERSION}")
