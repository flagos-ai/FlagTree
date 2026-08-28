cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Set LLVM_SYSPATH from environment variable
if(NOT DEFINED LLVM_SYSPATH)
  if(DEFINED ENV{LLVM_SYSPATH})
    set(LLVM_SYSPATH $ENV{LLVM_SYSPATH})
  else()
    message(FATAL_ERROR "LLVM_SYSPATH environment variable is not defined")
  endif()
endif()

# Project name and version
project(Profiler LANGUAGES CXX C)

# Define standard include directories
include_directories(${LLVM_SYSPATH}/include/)

# Set build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

# Collect all source files from the vendor directory
file(GLOB_RECURSE SOURCES ./*.cpp)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER ${LLVM_SYSPATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${LLVM_SYSPATH}/bin/clang++)

# Add the library target
set(BIN_NAME tx-profiler)
add_executable(${BIN_NAME} ${SOURCES})

# Apply RISC-V specific settings to our target
set(COMPILE_OPTIONS
  -fPIC
  -fno-rtti
  --std=c++17
  -O2
)
target_compile_options(${BIN_NAME} PRIVATE ${COMPILE_OPTIONS})

# target_link_directories(${BIN_NAME} ${LLVM_SYSPATH}/lib)
target_link_libraries(${BIN_NAME} PRIVATE
  LLVMIRReader
  LLVMAsmParser
  LLVMBitReader
  LLVMCore
  LLVMBinaryFormat
  LLVMDemangle
  LLVMRemarks
  LLVMBitstreamReader
  LLVMSupport
  LLVMTargetParser
)

# Set properties for the library
set_target_properties(${BIN_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# Install targets
install(TARGETS ${BIN_NAME}
  RUNTIME DESTINATION ${INSTALL_TSINGMICRO_DIR}/bin
)
