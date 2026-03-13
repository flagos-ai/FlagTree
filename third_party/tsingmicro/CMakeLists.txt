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

if(NOT DEFINED NO_INTRNISIC_RUN)
  if(DEFINED ENV{NO_INTRNISIC_RUN})
    set(NO_INTRNISIC_RUN $ENV{NO_INTRNISIC_RUN})
  endif()
endif()

if(NOT DEFINED ENABLE_SYNCHRONOUS_INTRINSIC)
  if(DEFINED ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
    set(ENABLE_SYNCHRONOUS_INTRINSIC $ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XUANTIE_NAME Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2)
set(INSTALL_TSINGMICRO_DIR ${CMAKE_INSTALL_PREFIX}/triton/backends/tsingmicro/)
install(CODE "file(MAKE_DIRECTORY \"${INSTALL_TSINGMICRO_DIR}\")")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/crt/include)
include_directories(${TX8_DEPS_ROOT}/include)

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
