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

if(NOT DEFINED XUANTIE_DIR)
  if(DEFINED ENV{XUANTIE_DIR})
    set(XUANTIE_DIR $ENV{XUANTIE_DIR})
  else()
    message(FATAL_ERROR "XUANTIE_DIR environment variable is not defined")
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/${TARGET})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${TX8_DEPS_ROOT}/tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/)

# Set build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

# Library name: vr stands for Vendor Runtime
set(VENDOR_RUNTIME_LIB vr)

# Collect all source files from the vendor directory
file(GLOB_RECURSE VENDOR_SOURCES lib/${TARGET}/*.c)

set(CMAKE_SYSTEM_NAME Generic)

if (USE_SIM_MODE)
  set(CMAKE_C_COMPILER ${LLVM_SYSPATH}/bin/clang)
  set(CMAKE_CXX_COMPILER ${LLVM_SYSPATH}/bin/clang++)
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
  set(CMAKE_C_COMPILER ${XUANTIE_DIR}/bin/riscv64-unknown-elf-gcc)
  set(CMAKE_CXX_COMPILER ${XUANTIE_DIR}/bin/riscv64-unknown-elf-g++)
  # set(CMAKE_C_COMPILER ${LLVM_SYSPATH}/bin/clang)
  # set(CMAKE_CXX_COMPILER ${LLVM_SYSPATH}/bin/clang++)
  # Define RISC-V target triple
  set(RISCV_TRIPLE "riscv64-unknown-elf")
  set(CMAKE_SYSTEM_PROCESSOR riscv)
  set(RISCV_STRIP ${XUANTIE_DIR}/bin/${RISCV_TRIPLE}-strip)

  include_directories(${XUANTIE_DIR}/riscv64-unknown-elf/include)
  # include_directories(${TX8_DEPS_ROOT}/include)
  # include_directories(${TX8_DEPS_ROOT}/include/bsp/xuantie_riscv_tx81/board_riscv_tx81/include)
  # include_directories(${TX8_DEPS_ROOT}/include/rtthread/include)

  # Define RISC-V specific compile options
  set(RISCV_COMPILE_OPTIONS
    -mcpu=c908
    # clang config
    # --target=${RISCV_TRIPLE}
    # -march=rv64imfdc
    # -march=rv64imafdc_xtheadcmo_xtheadsync
    # -mabi=lp64d
    -mcmodel=medany
    -DCONFIG_TX8_KERNEL_PRINTF_SUPPORT=1
  )

  if (ENABLE_PROFILING)
    # 只需要定义，不需要链接，编译kernel.so的时候链接profile
    set(RISCV_COMPILE_OPTIONS ${RISCV_COMPILE_OPTIONS}
      -DENABLE_PROFILING
    )
  endif()

  if (ENABLE_NO_INTRINSIC_RUN)
    set(RISCV_COMPILE_OPTIONS ${RISCV_COMPILE_OPTIONS}
      -DENABLE_NO_INTRINSIC_RUN
    )
  endif()

  if (ENABLE_SYNCHRONOUS_INTRINSIC)
    set(RISCV_COMPILE_OPTIONS ${RISCV_COMPILE_OPTIONS}
      -DENABLE_SYNCHRONOUS_INTRINSIC
    )
  endif()

  if (ENABLE_DEBUG_DUMP_DATA)
    set(RISCV_COMPILE_OPTIONS ${RISCV_COMPILE_OPTIONS}
      -DENABLE_DEBUG_DUMP_DATA
    )
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RISCV_COMPILE_OPTIONS ${RISCV_COMPILE_OPTIONS}
      -DRT_USING_DEBUG
    )
  endif()

  # Add the library target
  add_library(${VENDOR_RUNTIME_LIB} STATIC ${VENDOR_SOURCES})

  target_link_libraries(${VENDOR_RUNTIME_LIB} PRIVATE rcsfw_include_interface)

  # Apply RISC-V specific settings to our target
  target_compile_options(${VENDOR_RUNTIME_LIB} PRIVATE ${RISCV_COMPILE_OPTIONS})
  target_link_options(${VENDOR_RUNTIME_LIB} PRIVATE --target=${RISCV_TRIPLE})

  # Set properties for the library
  set_target_properties(${VENDOR_RUNTIME_LIB} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    SUFFIX ".a"
  )

  add_custom_command(TARGET ${VENDOR_RUNTIME_LIB} POST_BUILD
    COMMAND ${RISCV_STRIP} --strip-debug $<TARGET_FILE:${VENDOR_RUNTIME_LIB}>
    VERBATIM
  )

endif()

install(TARGETS ${VENDOR_RUNTIME_LIB}
  ARCHIVE DESTINATION ${INSTALL_TSINGMICRO_DIR}/lib
)

option(BUILD_TX81_HOST_TESTS "Build Tx81 host unit tests" OFF)
if(BUILD_TX81_HOST_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
