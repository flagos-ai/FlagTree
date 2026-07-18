# =============================================================================
# Triton GCU300 Build Configuration
# =============================================================================
#
# This file controls the build flow for GCU300 architecture by calling
# functions defined in cmake/triton_gcu.cmake.
#
# Build steps:
#   1. Setup LLVM/MLIR for Triton
#   2. Fetch and build Triton source
#   3. Build core library (lib_triton_gcu300_core.so)
#   4. Build tools (triton-gcu300-opt)
#   5. Build Python bindings (_triton_gcu300.so)
#   6. Setup unit tests
#
# =============================================================================

set(arch "gcu300")



# Load function library
include(triton_gcu)

# -----------------------------------------------------------------------------
# Step 1: LLVM/MLIR Setup (organized by commit, shared across architectures)
# -----------------------------------------------------------------------------
# Load setup LLVM function
setup_llvm_download("${TRITON_GCU300_LLVM_HASH}" KURAMA_LLVM_DIR)

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
message(STATUS "[GCU300] mlir-tblgen: ${MLIR_TABLEGEN_EXE}")

# Compiler flags for GCU300
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
include(triton_gcu300)

# Fetch and build Triton
setup_triton_fetch(
  TRITON_COMMIT "${TRITON_GCU300_HASH}"
  MLIR_DIR "${MLIR_DIR}"
  LLVM_LIBRARY_DIR "${LLVM_LIBRARY_DIR}"
  GIT_URL_DIR "${project_git_url_dir}"
  MLIR_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}"
  OBJECT_FILES "${triton_gcu300_objs}"
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

add_library(triton_gcu300 INTERFACE)

target_include_directories(triton_gcu300 SYSTEM INTERFACE ${TRITON_SOURCE_DIR}/include)
target_include_directories(triton_gcu300 SYSTEM INTERFACE ${TRITON_BINARY_DIR}/include)
target_include_directories(triton_gcu300 SYSTEM INTERFACE ${TRITON_SOURCE_DIR})
target_include_directories(triton_gcu300 SYSTEM INTERFACE ${TRITON_BINARY_DIR}/lib/Dialect/Triton/Transforms)
if(TARGET ${TRITON_BUILD_TARGET})
  add_dependencies(triton_gcu300 ${TRITON_BUILD_TARGET})
endif()

# -----------------------------------------------------------------------------
# Step 4: Build Dialects and Libraries
# -----------------------------------------------------------------------------
add_subdirectory(include)
add_subdirectory(lib)

# -----------------------------------------------------------------------------
# Step 5: Build Tools (triton-gcu300-opt)
# -----------------------------------------------------------------------------
# Standalone tool using only the C API from triton_gcu300_core
add_executable(triton-gcu300-opt triton-gcu300-opt.cpp)

set_target_properties(triton-gcu300-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH "$ORIGIN/../lib;$ORIGIN"
  INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN"
)

target_link_libraries(triton-gcu300-opt PRIVATE triton_gcu300_core)

# Install tools
install(TARGETS triton-gcu300-opt RUNTIME DESTINATION bin)

add_custom_target(triton-gcu300-tools ALL DEPENDS triton-gcu300-opt)
set(triton_gcu_source_dir ${CMAKE_CURRENT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# Step 6: Build Python Bindings
# -----------------------------------------------------------------------------
add_subdirectory(python)

# -----------------------------------------------------------------------------
# Step 7: Unit Tests
# -----------------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unittests)
  add_subdirectory(unittests)
endif()

# -----------------------------------------------------------------------------
# Export Version Information
# -----------------------------------------------------------------------------
set_property(GLOBAL PROPERTY TRITON_GCU300_VERSION ${TRITON_ORIG_VERSION})

message(STATUS "[gcu300] Build configuration complete")
message(STATUS "[gcu300] Triton version: ${TRITON_ORIG_VERSION}")
