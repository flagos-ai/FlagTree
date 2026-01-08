cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# FLAGTREE Options
set(FLAGTREE_BACKEND "$ENV{FLAGTREE_BACKEND}")
if(FLAGTREE_BACKEND)
    add_definitions(-DFLAGTREE_BACKEND=\"${FLAGTREE_BACKEND}\")
endif()

if(NOT FLAGTREE_BACKEND)
  add_definitions(-D__NVIDIA__)
  add_definitions(-D__AMD__)
elseif(FLAGTREE_BACKEND STREQUAL "iluvatar")
  add_definitions(-D__ILUVATAR__)
  remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
elseif(FLAGTREE_BACKEND STREQUAL "mthreads")
  set(ENV{PATH} "$ENV{LLVM_SYSPATH}/bin:$ENV{PATH}")
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(ENV{FLAGTREE_PLUGIN} $ENV{FLAGTREE_BACKEND})
elseif(FLAGTREE_BACKEND STREQUAL "aipu")
  set(CMAKE_C_COMPILER clang-16)
  set(CMAKE_CXX_COMPILER clang++-16)
  add_definitions(-D__NVIDIA__)
  add_definitions(-D__AMD__)
elseif(FLAGTREE_BACKEND STREQUAL "tsingmicro")
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
elseif(FLAGTREE_BACKEND STREQUAL "sunrise")
  # remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  # add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  find_package(Python3 3.10 REQUIRED COMPONENTS Development.Module Interpreter)
  if(EDITABLE_MODE)
    set (DEFAULT_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sunrise")
  else()
    set (DEFAULT_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
  endif()
  add_definitions(-DDEFAULT_PLUGIN_DIR="${DEFAULT_PLUGIN_DIR}")
endif()
set(FLAGTREE_PLUGIN "$ENV{FLAGTREE_PLUGIN}")
if(FLAGTREE_PLUGIN)
  add_definitions(-D__FLAGTREE_PLUGIN__)
endif()

# FLAGTREE SPEC LIB GET FUNC
function(get_flagtree_backend_lib lib_name output_lib)
  set(ret FlagTree_${FLAGTREE_BACKEND}_${lib_name})
  if(NOT TARGET ${ret})
    set(ret "")
  endif()
  set(${output_lib} ${ret} PARENT_SCOPE)
endfunction()

# FLAGTREE SPEC TD FILE GET FUNC
function(set_flagtree_backend_td output_td td_filename)
  set(ret ${td_filename})
  file(RELATIVE_PATH relative_path "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  get_filename_component(BACKEND_SPEC_ROOT "${BACKEND_SPEC_INCLUDE_DIR}" DIRECTORY)
  set(BACKEND_SPEC_TD ${BACKEND_SPEC_ROOT}/${relative_path}/${td_filename})
  if(EXISTS ${BACKEND_SPEC_TD})
    set(ret ${BACKEND_SPEC_TD})
  endif()
  set(${output_td} ${ret} PARENT_SCOPE)
endfunction()

project(triton CXX C)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)
if(FLAGTREE_BACKEND)
  option(TRITON_BUILD_PROTON "Build the Triton Proton profiler" OFF)
  option(TRITON_BUILD_UT "Build C++ Triton Unit Tests" OFF)
else()
  option(TRITON_BUILD_PROTON "Build the Triton Proton profiler" ON)
  option(TRITON_BUILD_UT "Build C++ Triton Unit Tests" ON)
endif()
option(TRITON_BUILD_WITH_CCACHE "Build with ccache (if available)" ON)
set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "Enable different codegen backends")

if(TRITON_BUILD_WITH_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "CXX compiler launcher")
  else()
    message(
      STATUS
        "Could not find ccache. Consider installing ccache to speed up compilation."
    )
  endif()
endif()

set(TRITON_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if (TRITON_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${TRITON_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
endif()


# Ensure Python3 vars are set correctly
# used conditionally in this file and by lit tests

# Customized release build type with assertions: TritonRelBuildWithAsserts
if(NOT MSVC)
  set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
  set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
  set(CMAKE_C_FLAGS_TRITONBUILDWITHO1 "-O1")
  set(CMAKE_CXX_FLAGS_TRITONBUILDWITHO1 "-O1")
else()
  set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "/Zi /RTC1 /bigobj /Zc:preprocessor /permissive-")
  set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "/Zi /RTC1 /bigobj /Zc:preprocessor /permissive-")
  set(CMAKE_EXE_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
  set(CMAKE_MODULE_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
  set(CMAKE_SHARED_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT WIN32)
  find_library(TERMINFO_LIBRARY tinfo)
endif()

if(TRITON_BUILD_UT)
  # This is an aggregate target for all unit tests.
  add_custom_target(TritonUnitTests)
  set_target_properties(TritonUnitTests PROPERTIES FOLDER "Triton/Tests")
  include(AddTritonUnitTest)
endif()

# Compiler flags
set(FLAGTREE_BACKEND_DIR ${PROJECT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND})
## flagtree spec include dir
set(BACKEND_SPEC_INCLUDE_DIR ${FLAGTREE_BACKEND_DIR}/backend/spec/include)
if(FLAGTREE_BACKEND AND EXISTS ${BACKEND_SPEC_INCLUDE_DIR})
  include_directories(${BACKEND_SPEC_INCLUDE_DIR})
endif()
## flagtree third_party include dir
set(BACKEND_INCLUDE_DIR ${FLAGTREE_BACKEND_DIR}/include)
if(FLAGTREE_BACKEND AND EXISTS "${BACKEND_INCLUDE_DIR}")
  include_directories(${BACKEND_INCLUDE_DIR})
else()
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")
endif()


# #########
# LLVM
# #########
if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Utilities
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )


  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_triton_object)

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  cmake_parse_arguments(ARG "" "SHARED_LIB" "LINK_LIBS" ${ARGN})
  if(ARG_SHARED_LIB)
    set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${ARG_SHARED_LIB})
  else()
    set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
    add_triton_object(${name} ${ARGN})
  endif()
endfunction()


# Disable warnings that show up in external code (gtest;pybind11)
if(NOT MSVC)
  if(FLAGTREE_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4624 /wd4715 /wd4530")
endif()

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
if(FLAGTREE_BACKEND AND EXISTS ${BACKEND_INCLUDE_DIR})
  include_directories(${PROJECT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND}/include)
  include_directories(${PROJECT_BINARY_DIR}/third_party/${FLAGTREE_BACKEND}/include) # Tablegen'd files
else()
  include_directories(${PROJECT_SOURCE_DIR}/include)
  include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
endif()
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party) # Tablegen'd files

# link_directories(${LLVM_LIBRARY_DIR})
if (FLAGTREE_BACKEND MATCHES "^(cambricon|aipu|tsingmicro)$")
  include_directories(${PROJECT_SOURCE_DIR}/include)
  include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
  add_subdirectory(include)
  add_subdirectory(lib)
elseif(NOT FLAGTREE_BACKEND)
  add_subdirectory(include)
  add_subdirectory(lib)
endif()

# TODO: Figure out which target is sufficient to fix errors; triton is
# apparently not enough. Currently set linking libstdc++fs for all targets
# to support some old version GCC compilers like 8.3.0.
if (NOT WIN32 AND NOT APPLE AND NOT BSD)
  link_libraries(stdc++fs)
endif()


# -----

# ------
if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND}/python/src)
  if(NOT (FLAGTREE_BACKEND AND EXISTS "${PYTHON_SRC_PATH}"))
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  endif()
  include_directories(${PYTHON_SRC_PATH})

  # Python Interpreter is used to run lit tests
  find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${pybind11_INCLUDE_DIR})
  link_directories(${Python3_LIBRARY_DIRS})
  link_libraries(${Python3_LIBRARIES})
  add_link_options(${Python3_LINK_OPTIONS})

  if (DEFINED TRITON_PLUGIN_DIRS)
    foreach(PLUGIN_DIR ${TRITON_PLUGIN_DIRS})
      # Read the plugin name under dir/backend/name.conf
      cmake_path(APPEND PLUGIN_DIR "backend" "name.conf" OUTPUT_VARIABLE PLUGIN_NAME_PATH)
      file(READ ${PLUGIN_NAME_PATH} PLUGIN_NAME)
      string(STRIP ${PLUGIN_NAME} PLUGIN_NAME)

      list(APPEND TRITON_PLUGIN_NAMES ${PLUGIN_NAME})

      # Include the plugin as part of the build, placing the build output under
      # ${TRITON_BINARY_DIR}/third_party/${PLUGIN_NAME}
      cmake_path(APPEND TRITON_BINARY_DIR "third_party" ${PLUGIN_NAME} OUTPUT_VARIABLE PLUGIN_DIR_BUILD_OUTPUT)
      message(STATUS "Building plugin '${PLUGIN_NAME}' from ${PLUGIN_DIR} with output ${PLUGIN_DIR_BUILD_OUTPUT}")
      add_subdirectory(${PLUGIN_DIR} ${PLUGIN_DIR_BUILD_OUTPUT})
    endforeach()
  endif()

  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()

  if (TRITON_BUILD_PROTON)
    add_definitions(-D__PROTON__)
    add_subdirectory(third_party/proton)
  endif()
  # We always build proton dialect
  list(APPEND TRITON_PLUGIN_NAMES "proton")
  add_subdirectory(third_party/proton/dialect)

  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
  get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)
  set(TRITON_LIBRARIES
    ${triton_libs}
    ${triton_plugins}

    # mlir
    MLIRAMDGPUDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUTransforms
    MLIRIR
    MLIRControlFlowToLLVM
    MLIRBytecodeWriter
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRMathToLLVM
    MLIRROCDLToLLVMIRTranslation
    MLIRGPUDialect
    MLIRSCFToControlFlow
    MLIRIndexToLLVM
    MLIRGPUToROCDLTransforms
    MLIRUBToLLVM

    # LLVM
    LLVMPasses
    LLVMNVPTXCodeGen
    # LLVMNVPTXAsmPrinter
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser

    Python3::Module
    pybind11::headers

  )
  if(FLAGTREE_BACKEND STREQUAL "iluvatar")
    set(TRITON_LIBRARIES
      ${triton_libs}
      ${triton_plugins}

      # mlir
      MLIRNVVMDialect
      MLIRNVVMToLLVMIRTranslation
      MLIRGPUToNVVMTransforms
      MLIRGPUToGPURuntimeTransforms
      MLIRGPUTransforms
      MLIRIR
      MLIRControlFlowToLLVM
      MLIRBytecodeWriter
      MLIRPass
      MLIRTransforms
      MLIRLLVMDialect
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRMathToLLVM
      MLIRGPUDialect
      MLIRSCFToControlFlow
      MLIRIndexToLLVM

      # LLVM
      LLVMPasses
      LLVMIluvatarCodeGen
      LLVMIluvatarAsmParser
    )
  elseif(FLAGTREE_BACKEND STREQUAL "xpu")
    set(TRITON_LIBRARIES
      ${triton_libs}
      ${triton_plugins}

      # mlir
      MLIRIR
      MLIRControlFlowToLLVM
      MLIRBytecodeWriter
      MLIRPass
      MLIRTransforms
      MLIRLLVMDialect
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRMathToLLVM
      MLIRGPUDialect
      MLIRSCFToControlFlow
      MLIRIndexToLLVM

      # LLVM
      LLVMPasses
      LLVMXPUCodeGen
      LLVMXPUAsmParser
    )
  elseif(FLAGTREE_BACKEND STREQUAL "tsingmicro")
    list(APPEND TRITON_LIBRARIES
	  # riscv
      LLVMRISCVCodeGen
      LLVMRISCVAsmParser
    )
  elseif(FLAGTREE_BACKEND STREQUAL "sunrise")
    set(TRITON_LIBRARIES
      ${triton_libs}
      ${triton_plugins}

      # mlir
      # MLIRAMDGPUDialect
      # MLIRNVVMDialect
      MLIRSTVMDialect  # STVM
      MLIRNVVMToLLVMIRTranslation
      MLIRSTVMToLLVMIRTranslation
      MLIRGPUToNVVMTransforms
      MLIRGPUToSTVMTransforms
      MLIRGPUToGPURuntimeTransforms
      MLIRGPUTransforms
      MLIRIR
      MLIRControlFlowToLLVM
      MLIRBytecodeWriter
      MLIRPass
      MLIRTransforms
      MLIRLLVMDialect
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRMathToLLVM
      # MLIRROCDLToLLVMIRTranslation
      MLIRGPUDialect
      MLIRSCFToControlFlow
      MLIRIndexToLLVM
      MLIRGPUToROCDLTransforms
      MLIRUBToLLVM

      # LLVM
      LLVMPasses
      # LLVMNVPTXCodeGen
      # LLVMAMDGPUCodeGen
      # LLVMAMDGPUAsmParser
      LLVMSTCUCodeGen
      LLVMSTCUAsmParser
      LLVMAArch64CodeGen
      LLVMAArch64AsmParser
      LLVMRISCVCodeGen
      LLVMRISCVAsmParser

      Python3::Module
      pybind11::headers
  )

  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR # Linux arm64
     CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR # macOS arm64
     CMAKE_OSX_ARCHITECTURES MATCHES "arm64")  # also macOS arm64
      list(APPEND TRITON_LIBRARIES
          LLVMAArch64CodeGen
          LLVMAArch64AsmParser
      )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
      list(APPEND TRITON_LIBRARIES
          LLVMX86CodeGen
          LLVMX86AsmParser
      )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
      list(APPEND TRITON_LIBRARIES
        LLVMPowerPCAsmParser
        LLVMPowerPCCodeGen
      )
  else()
    message(FATAL_ERROR "LLVM codegen/ASM parser libs: This HW architecture (${CMAKE_SYSTEM_PROCESSOR}) is not configured in cmake lib dependencies.")
  endif()

  # Define triton library
  string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_CODEGEN_BACKENDS})

  if (DEFINED TRITON_PLUGIN_NAMES)
    string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_BACKENDS_TUPLE} ${TRITON_PLUGIN_NAMES})
  endif()

  message(STATUS "Triton backends tuple: ${TRITON_BACKENDS_TUPLE}")

  set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
  add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})
  if(FLAGTREE_BACKEND STREQUAL "cambricon")
    add_library(triton SHARED)
  else()
    add_library(triton SHARED ${PYTHON_SRC_PATH}/main.cc
                ${PYTHON_SRC_PATH}/ir.cc
                ${PYTHON_SRC_PATH}/gluon_ir.cc
                ${PYTHON_SRC_PATH}/passes.cc
                ${PYTHON_SRC_PATH}/interpreter.cc
                ${PYTHON_SRC_PATH}/llvm.cc)
  endif()

  # Link triton with its dependencies
  #target_link_libraries(triton PRIVATE ${TRITON_LIBRARIES})
  target_link_libraries(triton PUBLIC ${TRITON_LIBRARIES})
  if(WIN32)
    target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(triton PROPERTIES SUFFIX ".pyd")
    set_target_properties(triton PROPERTIES PREFIX "lib")
  else()
    target_link_libraries(triton PRIVATE z)
  endif()
  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})

  if (NOT DEFINED LLVM_SYSPATH)
      message(FATAL_ERROR "LLVM_SYSPATH must be set.")
  endif()

  if (NOT DEFINED TRITON_WHEEL_DIR)
      message(FATAL_ERROR "TRITON_WHEEL_DIR must be set.")
  endif()

  configure_file(
    "${LLVM_SYSPATH}/bin/FileCheck"
    "${TRITON_WHEEL_DIR}/FileCheck"
    COPYONLY)

endif()

if (UNIX AND NOT APPLE)
  #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--export-dynamic")
endif()

if(TRITON_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

  # Check if the platform is MacOS
  if(APPLE)
    set(PYTHON_LDFLAGS "-undefined dynamic_lookup")
  endif()

  target_link_options(triton PRIVATE ${PYTHON_LDFLAGS})
endif()

if(NOT TRITON_BUILD_PYTHON_MODULE)
  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()
  add_subdirectory(third_party/proton/dialect)
endif()

find_package(Threads REQUIRED)

add_subdirectory(third_party/f2reduce)

if(NOT FLAGTREE_BACKEND OR FLAGTREE_BACKEND MATCHES "^(aipu|tsingmicro)$")
  add_subdirectory(bin)
  add_subdirectory(test)
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
  # This target runs all the unit tests.
  add_custom_target(check-triton-unit-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS TritonUnitTests
    USES_TERMINAL
  )
endif()
