if(NOT DEFINED TX8_DEPS_ROOT)
  if(DEFINED ENV{TX8_DEPS_ROOT})
    set(TX8_DEPS_ROOT $ENV{TX8_DEPS_ROOT})
  else()
    message(FATAL_ERROR "TX8_DEPS_ROOT environment variable is not defined")
  endif()
endif()
# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

if(NOT DEFINED USE_HOST_PROFILE)
  if(DEFINED ENV{USE_HOST_PROFILE})
    set(USE_HOST_PROFILE $ENV{USE_HOST_PROFILE})
  endif()
endif()

if(NOT DEFINED ENABLE_PROFILING)
  if(DEFINED ENV{ENABLE_PROFILING})
    set(ENABLE_PROFILING $ENV{ENABLE_PROFILING})
  endif()
endif()

if(NOT DEFINED ENABLE_NO_INTRINSIC_RUN)
  if(DEFINED ENV{ENABLE_NO_INTRINSIC_RUN})
    set(ENABLE_NO_INTRINSIC_RUN $ENV{ENABLE_NO_INTRINSIC_RUN})
  endif()
endif()

if(NOT DEFINED ENABLE_SYNCHRONOUS_INTRINSIC)
  if(DEFINED ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
    set(ENABLE_SYNCHRONOUS_INTRINSIC $ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
  endif()
endif()

if(NOT DEFINED ENABLE_DEBUG_DUMP_DATA)
  if(DEFINED ENV{ENABLE_DEBUG_DUMP_DATA})
    set(ENABLE_DEBUG_DUMP_DATA $ENV{ENABLE_DEBUG_DUMP_DATA})
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RISC-V cross toolchain (Xuantie). No pinned version: the SDK declares which
# toolchain to use (cmake/sdk_info.cmake). Override with -DXUANTIE_DIR or the
# XUANTIE_DIR environment variable.
set(_xuantie_sdk ${TX8_DEPS_ROOT}/rcs1fw-rtt/tool/rcsfw-xuantie-sdk)

if(NOT DEFINED XUANTIE_DIR)
  if(DEFINED ENV{XUANTIE_DIR})
    set(XUANTIE_DIR $ENV{XUANTIE_DIR})
  else()
    if(EXISTS "${_xuantie_sdk}/cmake/sdk_info.cmake")
      file(STRINGS "${_xuantie_sdk}/cmake/sdk_info.cmake" _l
           REGEX "set\\(__DEFAULT_rcs_toolchain" LIMIT_COUNT 1)
      if(_l MATCHES "\"([^\"]+)\"")
        set(XUANTIE_DIR "${_xuantie_sdk}/${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()
endif()

if(NOT EXISTS "${XUANTIE_DIR}/bin/riscv64-unknown-elf-gcc")
  file(GLOB _avail LIST_DIRECTORIES true "${_xuantie_sdk}/Xuantie-*")
  message(FATAL_ERROR
    "RISC-V toolchain not found:\n"
    "  XUANTIE_DIR = '${XUANTIE_DIR}'\n"
    "  available   = ${_avail}\n"
    "  override    : export XUANTIE_DIR=<dir>  or  cmake -DXUANTIE_DIR=<dir>")
endif()
set(INSTALL_TSINGMICRO_DIR ${TRITON_WHEEL_DIR}/backends/tsingmicro/)
install(CODE "file(MAKE_DIRECTORY \"${INSTALL_TSINGMICRO_DIR}\")")

# Copy built runtime artifacts back into the source tree, so that running the
# tests after a plain `ninja` never picks up a stale binary. Disable with
# -DTSINGMICRO_SYNC_TO_SOURCE=OFF on read-only source trees.
option(TSINGMICRO_SYNC_TO_SOURCE "Sync built backend artifacts into the source tree" ON)
set(TSINGMICRO_BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backend)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/crt/include)

include_directories(${TX8_DEPS_ROOT}/rcs1fw-rtt/rcs1-intrisic/instr_rcs1/include)
include_directories(${TX8_DEPS_ROOT}/rcs1fw-rtt/rcs1-intrisic/common/include)
include_directories(${TX8_DEPS_ROOT}/rcs1fw-rtt/include/components/oplib_rcs1/riscv/riscv/include)
include(${TX8_DEPS_ROOT}/rcs1fw-rtt/rcsfw_include_interface.cmake)

include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/tools)
include_directories(${CMAKE_BINARY_DIR}/third_party/flir/include)

add_subdirectory(include)
add_subdirectory(lib)
option(TSINGMICRO_BUILD_BIN "Build tsingmicro-opt and related tools" ON)
if(TSINGMICRO_BUILD_BIN)
  add_subdirectory(bin)
endif()
add_subdirectory(crt)
add_subdirectory(profiler)

if(TRITON_BUILD_PYTHON_MODULE)
  # FIXME: Unify the libraries for TsingMicro into fewer ones
  add_triton_plugin(TritonTsingMicro ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_tsingmicro.cc
                    LINK_LIBS TritonSharedAnalysis TritonSharedAnalysisStructured MagicKernelIR
                    Tx81IR TritonTilingExtIR TritonStructuredIR TritonToCoreDialects
                    TritonToLinalg TritonToStructured StructuredToMemref LinalgToMagicKernel
                    TritonArithToLinalg CoreDialectsToMK Tx81ToLLVM AllocateSharedMemory ExportKernelSymbols
                    Tx81MemrefToLLVM MKToTx81 LLVMRISCVCodeGen LLVMRISCVAsmParser
                    MLIRFuncAllExtensions)
  target_link_libraries(TritonTsingMicro PRIVATE Python3::Module pybind11::headers)
endif()
if(TSINGMICRO_SYNC_TO_SOURCE)
  # Manual repair target: refreshes the source-tree copies without relinking
  # the (large) binaries. Use it after `git clean` or when switching branches.
  set(_tsm_sync_commands)
  list(APPEND _tsm_sync_commands
       COMMAND ${CMAKE_COMMAND} -E make_directory ${TSINGMICRO_BACKEND_SRC_DIR}/bin)
  list(APPEND _tsm_sync_commands
       COMMAND ${CMAKE_COMMAND} -E make_directory ${TSINGMICRO_BACKEND_SRC_DIR}/lib)
  if(TARGET tsingmicro-opt)
    list(APPEND _tsm_sync_commands
         COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 $<TARGET_FILE:tsingmicro-opt> ${TSINGMICRO_BACKEND_SRC_DIR}/bin/)
  endif()
  if(TARGET tx-profiler)
    list(APPEND _tsm_sync_commands
         COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 $<TARGET_FILE:tx-profiler> ${TSINGMICRO_BACKEND_SRC_DIR}/bin/)
  endif()
  if(TARGET vr)
    list(APPEND _tsm_sync_commands
         COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 $<TARGET_FILE:vr> ${TSINGMICRO_BACKEND_SRC_DIR}/lib/)
  endif()
  add_custom_target(backend-sync ${_tsm_sync_commands}
    COMMENT "Refreshing source-tree copies of the built backend artifacts"
    VERBATIM)
endif()

#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#endif()
#add_subdirectory(test)
