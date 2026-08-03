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
add_compile_options(-Werror)

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

set(XUANTIE_DIR ${TX8_DEPS_ROOT}/rcs1fw-rtt/tool/rcsfw-xuantie-sdk/Xuantie-900-gcc-elf-newlib-x86_64-V2.8.0)
set(INSTALL_TSINGMICRO_DIR ${CMAKE_INSTALL_PREFIX}/triton/backends/tsingmicro/)
install(CODE "file(MAKE_DIRECTORY \"${INSTALL_TSINGMICRO_DIR}\")")

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
add_subdirectory(bin)
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
#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#endif()
#add_subdirectory(test)
