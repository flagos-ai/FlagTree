cmake_minimum_required(VERSION 3.18)

set(TARGET Tx81)
# Set TARGET from environment variable
if(NOT DEFINED TARGET)
  if(DEFINED ENV{CRT_TARGET})
    set(TARGET $ENV{CRT_TARGET})
  else()
    message(FATAL_ERROR "CRT_TARGET environment variable is not defined")
  endif()
endif()

if(NOT DEFINED XUANTIE_NAME)
  if(DEFINED ENV{XUANTIE_NAME})
    set(XUANTIE_NAME $ENV{XUANTIE_NAME})
  else()
    message(FATAL_ERROR "XUANTIE_NAME environment variable is not defined")
  endif()
endif()

# Set LLVM_SYSPATH from environment variable
if(NOT DEFINED LLVM_SYSPATH)
  if(DEFINED ENV{LLVM_SYSPATH})
    set(LLVM_SYSPATH $ENV{LLVM_SYSPATH})
  else()
    message(FATAL_ERROR "LLVM_SYSPATH environment variable is not defined")
  endif()
endif()

if(NOT DEFINED TX8_DEPS_ROOT)
  if(DEFINED ENV{TX8_DEPS_ROOT})
    set(TX8_DEPS_ROOT $ENV{TX8_DEPS_ROOT})
  else()
    message(FATAL_ERROR "TX8_DEPS_ROOT environment variable is not defined")
  endif()
endif()

# Build for simulator or hardware
if(NOT DEFINED USE_SIM_MODE)
  if(DEFINED ENV{USE_SIM_MODE})
    set(USE_SIM_MODE $ENV{USE_SIM_MODE})
  else()
    set(USE_SIM_MODE OFF)
    message(STATUS "Building for hardware (USE_SIM_MODE not set)")
  endif()
endif()

# Project name and version
project(VendorRuntime LANGUAGES CXX C)

# Define standard include directories
include_directories(${TX8_DEPS_ROOT}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/${TARGET})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Set build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

# Library name: vr stands for Vendor Runtime
set(VENDOR_RUNTIME_LIB vr)

# Collect all source files from the vendor directory
file(GLOB_RECURSE VENDOR_SOURCES lib/${TARGET}/*.c)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER ${LLVM_SYSPATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${LLVM_SYSPATH}/bin/clang++)

if (USE_SIM_MODE)
  # Define simulator specific compile options
  set(SIMULATOR_COMPILE_OPTIONS
    -fPIC
    -DUSE_SIM_MODE
  )

  add_library(${VENDOR_RUNTIME_LIB} SHARED ${VENDOR_SOURCES})

  # Apply simulator specific settings to our target
  target_compile_options(${VENDOR_RUNTIME_LIB} PRIVATE ${SIMULATOR_COMPILE_OPTIONS})

  # Set properties for the library
  set_target_properties(${VENDOR_RUNTIME_LIB} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    SUFFIX ".so"
  )
else()
  # Define RISC-V target triple
  set(RISCV_TRIPLE "riscv64-unknown-elf")
  set(CMAKE_SYSTEM_PROCESSOR riscv)

  include_directories(${TX8_DEPS_ROOT}/include)
  include_directories(${TX8_DEPS_ROOT}/${XUANTIE_NAME}/riscv64-unknown-elf/include)

  # Define RISC-V specific compile options
  set(RISCV_COMPILE_OPTIONS
    --target=${RISCV_TRIPLE}
    -march=rv64imfdc
    -mabi=lp64d
    -mcmodel=medany
    -DCONFIG_TX8_KERNEL_PRINTF_SUPPORT=1
  )

  if (USE_PROFILE)
    # 只需要定义，不需要链接，编译kernel.so的时候链接profile
    set(RISCV_COMPILE_OPTIONS ${RISCV_COMPILE_OPTIONS}
      -DUSE_PROFILE
    )
  endif()

  # Add the library target
  add_library(${VENDOR_RUNTIME_LIB} STATIC ${VENDOR_SOURCES})

  # Apply RISC-V specific settings to our target
  target_compile_options(${VENDOR_RUNTIME_LIB} PRIVATE ${RISCV_COMPILE_OPTIONS})
  target_link_options(${VENDOR_RUNTIME_LIB} PRIVATE --target=${RISCV_TRIPLE})

  # Set properties for the library
  set_target_properties(${VENDOR_RUNTIME_LIB} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    SUFFIX ".a"
  )

endif()

install(TARGETS ${VENDOR_RUNTIME_LIB}
  ARCHIVE DESTINATION ${INSTALL_TSINGMICRO_DIR}/lib
)
