# =============================================================================
# Triton GCU400 Build Configuration
# =============================================================================
#
# This file controls the build flow for GCU400 architecture by calling
# functions defined in cmake/triton_gcu.cmake.
#
# Build steps:
#   1. Setup LLVM/MLIR for Triton
#   2. Fetch and build Triton source
#   3. Build core library (lib_triton_gcu400_core.so)
#   4. Build tools (triton-gcu400-opt)
#   5. Build Python bindings (_triton_gcu400.so)
#   6. Setup unit tests
#
# =============================================================================

set(arch "gcu400")

if(FLAGTREE_TLE)
  add_definitions(-DENABLE_TLE)
endif()

# Load function library
include(triton_gcu)

# -----------------------------------------------------------------------------
# Step 1: LLVM/MLIR Setup (organized by commit, shared across architectures)
# -----------------------------------------------------------------------------
# Setup LLVM download
setup_llvm_download("${TRITON_GCU400_LLVM_HASH}" KURAMA_LLVM_DIR)

# Find LLVM and MLIR packages
set(LLVM_ROOT_DIR ${KURAMA_LLVM_DIR})
set(LLVM_LIBRARY_DIR ${LLVM_ROOT_DIR}/lib)
set(LLVM_DIR ${LLVM_LIBRARY_DIR}/cmake/llvm)
set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)

find_package(LLVM REQUIRED HINTS ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(${LLVM_CMAKE_DIR}/TableGen.cmake)
include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)

# Find mlir-tblgen
find_program(MLIR_TABLEGEN_EXE mlir-tblgen
  HINTS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_ROOT_DIR}/bin
  NO_DEFAULT_PATH REQUIRED)
message(STATUS "[GCU400] mlir-tblgen: ${MLIR_TABLEGEN_EXE}")

# Compiler flags for GCU400
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter")
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)

# -----------------------------------------------------------------------------
# Step 2: Fetch and Build Triton (organized by commit, shared across architectures)
# -----------------------------------------------------------------------------
# Include arch-specific Triton object file list
include(triton_gcu400)

# Fetch and build Triton
setup_triton_fetch(
  TRITON_COMMIT "${TRITON_GCU400_HASH}"
  MLIR_DIR "${MLIR_DIR}"
  LLVM_LIBRARY_DIR "${LLVM_LIBRARY_DIR}"
  GIT_URL_DIR "${project_git_url_dir}"
  MLIR_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}"
  OBJECT_FILES "${triton_gcu400_objs}"
)

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

if(TARGET ${TRITON_BUILD_TARGET})
  add_dependencies(triton_gcu400 ${TRITON_BUILD_TARGET})
endif()
# -----------------------------------------------------------------------------
# Step 4: Build Dialects and Libraries
# -----------------------------------------------------------------------------
add_subdirectory(include)
add_subdirectory(lib)

# -----------------------------------------------------------------------------
# Step 5: Build Tools (triton-gcu400-opt)
# -----------------------------------------------------------------------------
# Standalone tool using only the C API from triton_gcu400_core
add_executable(triton-gcu400-opt triton-gcu400-opt.cpp)

set_target_properties(triton-gcu400-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH "$ORIGIN/../lib;$ORIGIN"
  INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN"
)

target_link_libraries(triton-gcu400-opt PRIVATE triton_gcu400_core)

# Install tools
install(TARGETS triton-gcu400-opt RUNTIME DESTINATION bin)

add_custom_target(triton-gcu400-tools ALL DEPENDS triton-gcu400-opt)

# -----------------------------------------------------------------------------
# Step 6 : Build Python Bindings
# -----------------------------------------------------------------------------
add_subdirectory(python)

# -----------------------------------------------------------------------------
# Step 7: Unit Tests
# -----------------------------------------------------------------------------
set(triton_gcu_source_dir ${CMAKE_CURRENT_SOURCE_DIR})
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unittests)
  add_subdirectory(unittests)
endif()

# -----------------------------------------------------------------------------
# Export Version Information
# -----------------------------------------------------------------------------
set_property(GLOBAL PROPERTY TRITON_GCU400_VERSION ${TRITON_ORIG_VERSION})

message(STATUS "[gcu400] Build configuration complete")
message(STATUS "[gcu400] Triton version: ${TRITON_ORIG_VERSION}")
