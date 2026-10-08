# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_compile_definitions(__TRITON_VERSION_MAJOR__=3)
add_compile_definitions(__TRITON_VERSION_MINOR__=6)

macro(flagtree_configure_options)
  set(FLAGTREE_DEFAULT_OPTION ON)
  if(FLAGTREE_BACKEND)
    set(FLAGTREE_DEFAULT_OPTION OFF)
  endif()

  # CommonIR is an explicit opt-in for the default NVIDIA backend. Keep the
  # build contract as a C/C++ macro without introducing a CMake cache option.
  set(FLAGTREE_COMMON_IR_ENABLED "$ENV{FLAGTREE_COMMON_IR}")
  if(FLAGTREE_COMMON_IR_ENABLED)
    if(FLAGTREE_BACKEND)
      message(FATAL_ERROR "FLAGTREE_COMMON_IR requires the default NVIDIA backend")
    endif()
    add_compile_definitions(__FLAGTREE_COMMON_IR__)
  endif()

  set(FLAGCX_ENABLED OFF)
  set(FLAGCX_SUPPORT_BACKENDS nvidia iluvatar)
  if(NOT FLAGTREE_BACKEND OR
     "${FLAGTREE_BACKEND}" IN_LIST FLAGCX_SUPPORT_BACKENDS)
    add_compile_definitions(FLAGCX_ENABLED)
    set(FLAGCX_ENABLED ON)
  endif()

  set(FLAGTREE_TLE ON)
  if(FLAGTREE_BACKEND STREQUAL "xpu")
    set(FLAGTREE_TLE OFF)
  endif()
  if(FLAGTREE_TLE)
    add_definitions(-D__TLE__)
    list(APPEND LLVM_TABLEGEN_FLAGS -D__TLE__)
  endif()
  if(NOT FLAGTREE_BACKEND)
    add_definitions(-D__NVIDIA__)
    add_definitions(-D__AMD__)
    add_definitions(-D__FLAGTREE_REORDER_LOOP_LOADS__)
    add_definitions(-D__FLAGTREE_RLC_ENHANCE__)
    add_definitions(-D__FLAGTREE_SAME_WARP_LAYOUT_SHUFFLE__)
    add_definitions(-D__FLAGTREE_CONCAT_DOT_OPERAND__)
    list(APPEND LLVM_TABLEGEN_FLAGS -D__FLAGTREE_CONCAT_DOT_OPERAND__)
  elseif(FLAGTREE_BACKEND STREQUAL "iluvatar")
    add_definitions(-D__ILUVATAR__)
    set(FLAGTREE_TLE OFF)
    set(FLAGTREE_ILUVATAR_TLE ON)
    add_definitions(-D__ILUVATAR_TLE__)
    remove_definitions(-D__TLE__)
    list(REMOVE_ITEM LLVM_TABLEGEN_FLAGS -D__TLE__)
  elseif(FLAGTREE_BACKEND STREQUAL "mthreads")
    set(ENV{PATH} "$ENV{LLVM_SYSPATH}/bin:$ENV{PATH}")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(FLAGTREE_TLE OFF)
    set(FLAGTREE_MTHREADS_TLE ON)
  elseif(FLAGTREE_BACKEND STREQUAL "aipu")
    set(CMAKE_C_COMPILER clang-16)
    set(CMAKE_CXX_COMPILER clang++-16)
    add_definitions(-D__NVIDIA__)
    add_definitions(-D__AMD__)
  elseif(FLAGTREE_BACKEND STREQUAL "tsingmicro")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
  elseif(FLAGTREE_BACKEND STREQUAL "hcu")
    add_definitions(-D__HCU__)
  elseif(FLAGTREE_BACKEND STREQUAL "metax")
    add_definitions(-DUSE_MACA)
    option(BUILD_MCTLE "use maca triton language extensions" ON)
    if(BUILD_MCTLE)
      list(APPEND TRITON_PLUGIN_NAMES "mctle")
      add_definitions(-D__MCTLE__)
      # The .td files guard their mctle parts with #ifdef __MCTLE__ too
      # (TritonOps.td's atomic_rmw / atomic_cas pointer constraint and
      # shared-memory effects), and add_definitions does not reach mlir-tblgen.
      list(APPEND LLVM_TABLEGEN_FLAGS -D__MCTLE__)
    endif()
    set(FLAGTREE_TLE OFF)
    remove_definitions(-D__TLE__)
    list(REMOVE_ITEM LLVM_TABLEGEN_FLAGS -D__TLE__)
  elseif(FLAGTREE_BACKEND STREQUAL "sunrise")
    find_package(Python3 3.10 REQUIRED COMPONENTS Development.Module Interpreter)
  elseif(FLAGTREE_BACKEND STREQUAL "ppu")
    add_definitions(-D__PPU__)
  endif()

  set(FLAGTREE_PLUGIN "$ENV{FLAGTREE_PLUGIN}")
  if(FLAGTREE_PLUGIN)
    add_definitions(-D__FLAGTREE_PLUGIN__)
  endif()
endmacro()


# FlagPrism: configure the external profiler/debugger after base options exist.
macro(flagtree_configure_flagprism)
  set(_flagprism_default OFF)
  # FlagPrism: enable the external tools for the supported mthreads backend.
  if(FLAGTREE_BACKEND MATCHES "^(ascend|iluvatar|mthreads)$")
    set(_flagprism_default ON)
  endif()
  option(TRITON_BUILD_FLAGPRISM
         "Build the FlagPrism debugger and profiler"
         ${_flagprism_default})

  if(TRITON_BUILD_FLAGPRISM)
    # FlagPrism: accept mthreads as a supported integration backend.
    if(NOT FLAGTREE_BACKEND MATCHES "^(ascend|iluvatar|mthreads)$")
      message(FATAL_ERROR
        "TRITON_BUILD_FLAGPRISM is only supported when "
        "FLAGTREE_BACKEND is ascend, iluvatar, or mthreads.")
    endif()
    if(TRITON_BUILD_PROTON)
      message(FATAL_ERROR
        "TRITON_BUILD_FLAGPRISM and TRITON_BUILD_PROTON cannot both be enabled. "
        "Select exactly one profiler implementation.")
    endif()

    # FlagPrism: permit a separate local checkout during backend development.
    if(NOT FLAGPRISM_SOURCE_DIR)
      set(FLAGPRISM_SOURCE_DIR
          "${CMAKE_CURRENT_SOURCE_DIR}/third_party/FlagPrism")
    endif()
    # FlagPrism: resolve relative overrides against the FlagTree source root.
    get_filename_component(FLAGPRISM_SOURCE_DIR "${FLAGPRISM_SOURCE_DIR}" ABSOLUTE
                           BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(FLAGPRISM_CMAKE_FILE "${FLAGPRISM_SOURCE_DIR}/cmake/FlagPrism.cmake")
    if(EXISTS "${FLAGPRISM_CMAKE_FILE}")
      include("${FLAGPRISM_CMAKE_FILE}")
      add_compile_definitions(__FLAGPRISM__=1)
    else()
      message(FATAL_ERROR
        "FlagPrism source is missing. Run the Python package build to download "
        "third-party dependencies.")
    endif()
  endif()
endmacro()

# FlagPrism: register external profiler/debugger component targets.
macro(flagtree_add_flagprism_components)
  if(TRITON_BUILD_FLAGPRISM)
    flagprism_add_components()
  endif()
endmacro()


macro(flagtree_configure_codegen_backends)
  if(FLAGTREE_BACKEND STREQUAL "metax")
    list(APPEND TRITON_CODEGEN_BACKENDS "nvidia")
    list(APPEND TRITON_CODEGEN_BACKENDS "amd")
  endif()
endmacro()


macro(flagtree_include_directories root_include_dir)
  set(FLAGTREE_BACKEND_DIR ${PROJECT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND})

  # flagtree spec include dir
  set(BACKEND_SPEC_INCLUDE_DIR ${FLAGTREE_BACKEND_DIR}/spec_cpp/include)
  if(FLAGTREE_BACKEND AND EXISTS ${BACKEND_SPEC_INCLUDE_DIR})
    include_directories(${BACKEND_SPEC_INCLUDE_DIR})
  endif()

  # flagtree third_party include dir
  set(BACKEND_INCLUDE_DIR ${FLAGTREE_BACKEND_DIR}/include)
  if(FLAGTREE_BACKEND AND EXISTS "${BACKEND_INCLUDE_DIR}")
    # XPU backend Analysis headers (Utility.h, etc.) are structurally different
    # from the main-tree headers (not a superset), so core lib compilation needs
    # the main-tree include as well. XPU's own sub-cmake uses
    # include_directories(BEFORE ...) to ensure XPU targets pick up the XPU
    # versions first. Other backends provide superset headers and do not need
    # this.
    if(FLAGTREE_BACKEND STREQUAL "xpu")
      include_directories("${root_include_dir}")
    endif()
    include_directories(${BACKEND_INCLUDE_DIR})
  else()
    include_directories("${root_include_dir}")
  endif()
endmacro()


macro(flagtree_configure_backend_cxx_flags)
  if(FLAGTREE_BACKEND MATCHES "^(enflame|hcu|rpu|thrive|metax|xpu|tileir|ppu|spacemit)$")
    # Suppress visibility warnings in gluon_ir.cc (GCC 13+ -Wattributes on
    # pybind11 hidden types), and -Wcomment for generated
    # TritonGPUAttrDefs.h.inc (ASCII diagrams in TableGen output).
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Werror -Wno-attributes -Wno-comment -Wno-error=odr")
  elseif(FLAGTREE_BACKEND STREQUAL "iluvatar")
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -Wno-deprecated-declarations -Wno-attributes")
  elseif(FLAGTREE_BACKEND STREQUAL "mthreads")
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
  else()
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default")
  endif()
endmacro()


macro(flagtree_configure_core_source)
  if(BUILD_MCTLE)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/metax/plugin)
    include_directories(${PROJECT_BINARY_DIR}/third_party/metax/plugin)
  endif()

  if(FLAGTREE_BACKEND MATCHES
     "^(xpu|cambricon|aipu|tsingmicro|enflame|rpu|thrive|metax|tileir|ppu|spacemit|sunrise)$")
    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
    if(FLAGTREE_BACKEND STREQUAL "xpu")
      include_directories(${PROJECT_SOURCE_DIR}/third_party/nvidia/include)
      include_directories(${PROJECT_BINARY_DIR}/third_party/nvidia/include) # Tablegen'd files
      add_subdirectory(third_party/nvidia/include)
      include_directories(${PROJECT_SOURCE_DIR}/third_party/amd/include)
      include_directories(${PROJECT_BINARY_DIR}/third_party/amd/include) # Tablegen'd files
      add_subdirectory(third_party/amd/include)
    endif()
    add_subdirectory(include)
    add_subdirectory(lib)
    if(FLAGTREE_BACKEND STREQUAL "xpu")
      add_subdirectory(third_party/nvidia/lib/Dialect/NVGPU/IR)
      add_subdirectory(third_party/nvidia/lib/Dialect/NVWS/IR)
      add_subdirectory(third_party/nvidia/lib/Dialect/NVWS/Transforms)
      add_subdirectory(third_party/amd/lib/Dialect/TritonAMDGPU)
      foreach(_flagtree_xpu_core_target IN ITEMS
          TritonGPUTransforms)
        if(TARGET ${_flagtree_xpu_core_target})
          add_dependencies(${_flagtree_xpu_core_target}
            NVGPUTableGen
            NVGPUAttrDefsIncGen
            NVWSTableGen
            NVWSAttrDefsIncGen
            NVWSTransformsIncGen)
        endif()
      endforeach()
    endif()
  elseif(FLAGTREE_BACKEND STREQUAL "iluvatar")
    set(TRITON_CORE_SOURCE_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/iluvatar)
    set(TRITON_CORE_BINARY_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/third_party/iluvatar)
    include_directories(${TRITON_CORE_SOURCE_DIR}/include)
    include_directories(${TRITON_CORE_BINARY_DIR}/include)
    include_directories(${TRITON_CORE_SOURCE_DIR}/backend/include)
    include_directories(${TRITON_CORE_BINARY_DIR}/backend/include)
    if(FLAGTREE_ILUVATAR_TLE)
      include_directories(${TRITON_CORE_SOURCE_DIR}/tle/dialect/include)
      include_directories(${TRITON_CORE_BINARY_DIR}/tle/dialect/include)
    endif()
    add_subdirectory(
      ${TRITON_CORE_SOURCE_DIR}/include ${TRITON_CORE_BINARY_DIR}/include)
    add_subdirectory(
      ${TRITON_CORE_SOURCE_DIR}/lib ${TRITON_CORE_BINARY_DIR}/lib)
  endif()
endmacro()


function(flagtree_add_distributed_plugin)
  if(TRITON_BUILD_PYTHON_MODULE AND
     FLAGTREE_BACKEND STREQUAL "hcu")
    message(STATUS "Adding Python module for TritonDistributed")
    set(PYTHON_DIST_SRC_PATH
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hcu/python/src/dist)
    add_triton_plugin(
      TritonDistributed
      ${PYTHON_DIST_SRC_PATH}/triton_distributed.cc
      ${PYTHON_DIST_SRC_PATH}/passes.cc
      ${PYTHON_DIST_SRC_PATH}/ir.cc)
    target_link_libraries(
      TritonDistributed
      PRIVATE DistributedIR SIMTIR ProtonIR Python3::Module pybind11::headers)
  endif()
endfunction()


macro(flagtree_python_src_path_set)
  set(BACKEND_PYTHON_SRC_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND}/python/src)
  include_directories(${BACKEND_PYTHON_SRC_PATH})
endmacro()


macro(flagtree_python_link_libraries)
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${pybind11_INCLUDE_DIR})
  link_directories(${Python3_LIBRARY_DIRS})
  link_libraries(${Python3_LIBRARIES})
  add_link_options(${Python3_LINK_OPTIONS})
endmacro()


macro(flagtree_configure_flir_dependency)
  if(FLAGTREE_COMMON_IR_ENABLED)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/flir/CMakeLists.txt")
      message(FATAL_ERROR "FLAGTREE_COMMON_IR requires third_party/flir")
    endif()
    include_directories(${PROJECT_SOURCE_DIR}/third_party/flir/include)
    include_directories(${PROJECT_BINARY_DIR}/third_party/flir/include)
    add_subdirectory(third_party/flir/include/mlir-ext/Dialect/CommonIR)
    add_subdirectory(third_party/flir/lib/Dialect/CommonIR)
  endif()

  if(FLAGTREE_BACKEND STREQUAL "tsingmicro")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/flir/CMakeLists.txt")
      message(FATAL_ERROR "The ${FLAGTREE_BACKEND} backend requires third_party/flir")
    endif()

    # TsingMicro only consumes FLIR's C++ targets; do not build its Python/CPU plugin.
    set(TRITON_SHARED_BUILD_CPU_BACKEND OFF)
    list(REMOVE_ITEM TRITON_CODEGEN_BACKENDS "flir")
    if(NOT TARGET TritonSharedUtils)
      add_subdirectory(
        "${PROJECT_SOURCE_DIR}/third_party/flir"
        "${PROJECT_BINARY_DIR}/third_party/flir"
      )
    endif()
  endif()
endmacro()


macro(flagtree_configure_tle_plugin append_tle_plugin)
  if(FLAGTREE_TLE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tle/CMakeLists.txt")
      if(${append_tle_plugin})
        list(APPEND TRITON_PLUGIN_NAMES "tle")
      endif()
      add_subdirectory(third_party/tle)
      flagtree_add_tle_generated_header_dependencies()
    endif()
  endif()
endmacro()


macro(flagtree_configure_python_plugins)
  # We always build proton dialect because core Triton conversion libraries link
  # ProtonIR. XPU-specific Proton GPU lowering wrappers are disabled inside the
  # Proton plugin instead of skipping the whole dialect.
  # FlagPrism: its external component supplies ProtonIR and dialect registration.
  # if(FLAGTREE_BACKEND STREQUAL "hcu")
  if(TRITON_BUILD_FLAGPRISM)
    # FlagPrism supplies the ProtonIR target and dialect registration.
  elseif(FLAGTREE_BACKEND STREQUAL "hcu")
    list(APPEND TRITON_PLUGIN_DIRS
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hcu/proton")
    include_directories(
      ${PROJECT_BINARY_DIR}/third_party/${FLAGTREE_BACKEND})
    add_subdirectory(third_party/hcu/proton/Dialect)
    add_subdirectory(third_party/nvidia)
  elseif(FLAGTREE_BACKEND STREQUAL "mthreads")
    include_directories(
      ${PROJECT_BINARY_DIR}/third_party/${FLAGTREE_BACKEND})
    add_subdirectory(third_party/mthreads/proton/Dialect)
  elseif(FLAGTREE_BACKEND STREQUAL "iluvatar")
    if(TRITON_BUILD_PROTON)
      list(APPEND TRITON_PLUGIN_NAMES "proton")
      add_subdirectory(third_party/proton/Dialect)
    endif()
  else()
    list(APPEND TRITON_PLUGIN_NAMES "proton")
    add_subdirectory(third_party/proton/Dialect)
  endif()
endmacro()


macro(flagtree_configure_backend_libraries)
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

      # NVIDIA compat (PTXAsmFormat for TritonInstrumentToLLVM)
      TritonNVIDIACompat
    )
    # XPU target libraries are private to XTDK; public LLVM lacks them. Reuse the
    # frontend probe from the top-level CMakeLists instead of linking them unconditionally.
    if(NOT _TRITON_PUBLIC_LLVM_FRONTEND)
      list(APPEND TRITON_LIBRARIES
        LLVMXPUCodeGen
        LLVMXPUAsmParser
      )
    endif()
  elseif(FLAGTREE_BACKEND STREQUAL "tsingmicro")
    list(APPEND TRITON_LIBRARIES
      # riscv
      LLVMRISCVCodeGen
      LLVMRISCVAsmParser
    )
  elseif(FLAGTREE_BACKEND STREQUAL "hcu")
    list(APPEND TRITON_PLUGIN_NAMES "distributed")
    add_subdirectory(test)
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
  elseif(FLAGTREE_BACKEND STREQUAL "spacemit")
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

      Python3::Module
      pybind11::headers

    )
  elseif(FLAGTREE_BACKEND STREQUAL "metax" AND BUILD_MCTLE)
    list(APPEND TRITON_LIBRARIES MLIRTargetLLVMIRImport)
  endif()
endmacro()


macro(flagtree_configure_shared_linker_flags)
  if(FLAGTREE_BACKEND STREQUAL "metax")
    file(GLOB _flagtree_llvm_archives CONFIGURE_DEPENDS
      "${LLVM_LIBRARY_PATH}/libLLVM*.a")
    foreach(_flagtree_llvm_archive ${_flagtree_llvm_archives})
      get_filename_component(
        _flagtree_llvm_archive_name "${_flagtree_llvm_archive}" NAME)
      set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,${_flagtree_llvm_archive_name}")
    endforeach()
  elseif(FLAGTREE_BACKEND STREQUAL "sunrise")
    set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--export-dynamic")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
  endif()
endmacro()


macro(flagtree_configure_tools_and_tests)
  if(NOT FLAGTREE_BACKEND OR
     FLAGTREE_BACKEND MATCHES
       "^(aipu|tsingmicro|enflame|rpu|thrive|metax|sunrise|tileir|ppu)$")
    add_subdirectory(bin)
    if(FLAGTREE_TLE)
      flagtree_add_tle_generated_header_dependencies()
    endif()
    add_subdirectory(test)
  elseif(FLAGTREE_BACKEND STREQUAL "iluvatar")
    option(FLAGTREE_ILUVATAR_BUILD_BIN
      "Build third_party/iluvatar/bin tools and lit tests" OFF)
    if(FLAGTREE_ILUVATAR_BUILD_BIN)
      add_subdirectory(
        ${TRITON_CORE_SOURCE_DIR}/bin ${TRITON_CORE_BINARY_DIR}/bin)
      add_subdirectory(test)
    endif()
  endif()

  # When PPU backend is built, the upstream TritonGPU transforms reference
  # PPU-specific dialect symbols (e.g. mlir::triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp)
  # that live in TritonPPUGPUIR (defined under third_party/ppu/lib/...). Wire that
  # dependency here, after both targets have been declared by their respective
  # add_subdirectory() calls above, so unittests/triton.so/triton-llvm-opt all link cleanly.
  if(FLAGTREE_BACKEND STREQUAL "ppu" AND TARGET TritonGPUTransforms AND TARGET TritonPPUGPUIR)
    target_link_libraries(TritonGPUTransforms PUBLIC TritonPPUGPUIR)
  endif()
endmacro()


function(flagtree_add_tle_generated_header_dependencies)
  if(NOT TARGET TleTableGen)
    return()
  endif()

  set(_flagtree_tle_codegen_deps TleTableGen)
  if(TARGET TritonTLETransformsIncGen)
    list(APPEND _flagtree_tle_codegen_deps TritonTLETransformsIncGen)
  endif()

  set(_flagtree_enflame_tle_header_targets
      MLIRTritonToGCU_gcu300
      MLIRTritonToGCU_gcu400
      MLIRGCUTritonToTritonGPU_gcu400
      MLIRTritonGCUTransforms_gcu400
      triton_gcu300_core
      triton_gcu400_core)

  # Native compiler targets include TLE generated headers under __TLE__ guards.
  # The TLE dialect is added after the core libraries, so the dependency must be
  # attached explicitly once the TLE tablegen targets exist; otherwise a clean
  # parallel build can compile those libraries before the generated .inc files.
  foreach(_flagtree_tle_header_target IN ITEMS
      TritonAnalysis
      TritonAMDAnalysis
      TritonToTritonGPU
      TritonGPUTransforms
      TritonNvidiaGPUTransforms
      TritonNVIDIAGPUToLLVM
      TritonGPUToLLVM
      NVHopperTransforms
      ${_flagtree_enflame_tle_header_targets}
      triton
      triton-opt
      triton-reduce
      triton-lsp
      triton-llvm-opt
      triton-tensor-layout)
    foreach(_flagtree_tle_dependency_target IN ITEMS
        ${_flagtree_tle_header_target}
        obj.${_flagtree_tle_header_target})
      if(TARGET ${_flagtree_tle_dependency_target})
        add_dependencies(${_flagtree_tle_dependency_target}
          ${_flagtree_tle_codegen_deps})
      endif()
    endforeach()
  endforeach()
endfunction()


macro(flagtree_added_python_src)
  set(_flagtree_python_sources)
  foreach(_flagtree_python_source ${ARGN})
    get_filename_component(_python_source_name "${_flagtree_python_source}" NAME)
    set(_python_source "${BACKEND_PYTHON_SRC_PATH}/${_python_source_name}")
    if(IS_DIRECTORY "${BACKEND_PYTHON_SRC_PATH}" AND EXISTS "${_python_source}")
      list(APPEND _flagtree_python_sources "${_python_source}")
    else()
      list(APPEND _flagtree_python_sources "${_flagtree_python_source}")
    endif()
  endforeach()
  add_library(triton SHARED ${_flagtree_python_sources})

  if(FLAGTREE_BACKEND STREQUAL "xpu")
    target_sources(triton PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xpu/python/src/mlir_pass_abi_shim.cc)
    add_dependencies(triton TritonAMDGPUTableGen TritonAMDGPUAttrDefsIncGen)
    target_link_libraries(triton PRIVATE TritonAMDGPUIR TritonAMDUtils)
  endif()
endmacro()


# FLAGTREE SPEC TD FILE GET FUNC
function(flagtree_spec_td_set output_td td_filename)
  set(ret ${td_filename})
  file(RELATIVE_PATH relative_path "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  get_filename_component(BACKEND_SPEC_ROOT "${BACKEND_SPEC_INCLUDE_DIR}" DIRECTORY)
  set(BACKEND_SPEC_TD ${BACKEND_SPEC_ROOT}/${relative_path}/${td_filename})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${BACKEND_SPEC_TD}")
  if(EXISTS ${BACKEND_SPEC_TD})
    set(ret ${BACKEND_SPEC_TD})
  endif()
  set(${output_td} ${ret} PARENT_SCOPE)
endfunction()


# ---------------------------------------------------------------------------
# Public LLVM 22 compatibility layer (XPU only)
# XPU builds against XTDK LLVM 22 and public LLVM 22; fork differences are not versioned.
# Probe headers with file(READ)+MATCHES: check_cxx_source_compiles misses MLIR include paths.
# Keep one macro per probe so changing one compatibility check cannot alter another.
#
# Scoped to the xpu backend: the flags below are global (CMAKE_CXX_FLAGS) and
# would otherwise leak into every other vendor's build.  Other backends keep
# their previous flags.
# ---------------------------------------------------------------------------
macro(flagtree_configure_backend_llvm_compat)
if(NOT MSVC AND FLAGTREE_BACKEND STREQUAL "xpu")
  # Derive the frontend from the LLVM tree in use: XTDKDLDialect.h identifies XTDK;
  # this avoids relying on a separately maintained external switch.
  set(_XDL_HDR "${LLVM_INCLUDE_DIRS}/mlir/Dialect/LLVMIR/XTDKDLDialect.h")
  if(EXISTS "${_XDL_HDR}")
    message(STATUS "[XPU] XTDKDL dialect found (XTDK tree)")
    add_compile_definitions(TRITON_HAVE_XTDKDL)
    set(_TRITON_PUBLIC_LLVM_FRONTEND OFF)
  else()
    message(STATUS "[XPU] XTDKDL dialect absent (public LLVM)")
    set(_TRITON_PUBLIC_LLVM_FRONTEND ON)
  endif()

  if(_TRITON_PUBLIC_LLVM_FRONTEND)
    # Public LLVM 22 deprecates mlir::OpBuilder::create<T>(); suppress the warning rather
    # than rewriting every call site shared with XTDK LLVM 22.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    # Clang 18+ warns about temporary values bound to local ValueRange/SmallVector
    # references; these bindings are safe. Probe the flag because GCC 11 rejects it.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Wno-error=dangling" CXX_SUPPORTS_WNO_ERROR_DANGLING)
    if(CXX_SUPPORTS_WNO_ERROR_DANGLING)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=dangling")
    endif()

    # Public LLVM 22's v2 tarball uses a GCC 11 static libstdc++ archive that references
    # symbols missing from the host GCC 9 archive. Link it after all LLVM archives and
    # derive its path from the LLVM tree in use instead of relying on an external override.
    if(EXISTS "${LLVM_SYSPATH}/lib/gcc11/libstdc++.a")
      set(_GCC11_LIBSTDCXX "${LLVM_SYSPATH}/lib/gcc11/libstdc++.a")
      message(STATUS "[XPU] gcc-11 static libstdc++ found: ${_GCC11_LIBSTDCXX}")
    endif()
  endif()

  # XTDK-private pipeline tuning options such as VectorCombineXPU.
  set(_PT_HDR "${LLVM_INCLUDE_DIRS}/llvm/Passes/PassBuilder.h")
  if(EXISTS "${_PT_HDR}")
    file(READ "${_PT_HDR}" _PT_SRC)
    if(_PT_SRC MATCHES "VectorCombineXPU")
      add_compile_definitions(TRITON_HAVE_XTDK_TUNING_OPTIONS)
    endif()
  endif()

  # Public MLIR 22 uses terminator-based RegionBranchPoint encoding and removed
  # getRegionOrNull; XTDK MLIR still uses the Region*-based form.
  set(_RBP_HDR "${LLVM_INCLUDE_DIRS}/mlir/Interfaces/ControlFlowInterfaces.h")
  if(EXISTS "${_RBP_HDR}")
    file(READ "${_RBP_HDR}" _RBP_SRC)
    if(_RBP_SRC MATCHES "getTerminatorPredecessorOrNull")
      message(STATUS "[XPU] RegionBranchPoint: terminator-based (public MLIR)")
      add_compile_definitions(TRITON_MLIR_REGIONBRANCHPOINT_TERMINATOR)
    endif()
  endif()

  # Public MLIR 22 provides a (Operation*, result_range) RegionSuccessor constructor;
  # XTDK 22.1.8 provides only the (result_range) form.
  set(_RS_HDR "${LLVM_INCLUDE_DIRS}/mlir/Interfaces/ControlFlowInterfaces.h")
  if(EXISTS "${_RS_HDR}")
    file(READ "${_RS_HDR}" _RS_SRC)
    if(_RS_SRC MATCHES "RegionSuccessor\\(Operation \\*successorOp")
      message(STATUS "[XPU] RegionSuccessor: (Operation*, result_range) ctor found (public MLIR)")
      add_compile_definitions(TRITON_MLIR_REGIONSUCCESSOR_OP_CTOR)
    else()
      message(STATUS "[XPU] RegionSuccessor: single-arg ctor (XTDK MLIR)")
      add_compile_definitions(TRITON_MLIR_REGIONSUCCESSOR_SINGLE_ARG)
    endif()
  else()
    add_compile_definitions(TRITON_MLIR_REGIONSUCCESSOR_OP_CTOR)
  endif()

  # XTDK's private printf/assert pass lowers device printf at LLVM O3, so MLIR must
  # preserve `call @printf`; public LLVM lowers it in MLIR instead. This gates MLIR
  # TargetInfo lowering only, distinct from llvm.cc's TRITON_HAVE_XPU_PRINTF_ASSERT probe.
  set(_PRINTF_HDR "${LLVM_INCLUDE_DIRS}/llvm/Transforms/Utils/XPULowerPrintfAssert.h")
  if(EXISTS "${_PRINTF_HDR}")
    message(STATUS "[XPU] XPU printf/assert pass found (XTDK tree): MLIR keeps call @printf")
  else()
    message(STATUS "[XPU] XPU printf/assert pass absent (public LLVM): MLIR lowers printf")
    add_compile_definitions(TRITON_XPU_PRINTF_MLIR_LOWER)
  endif()

  # Public MLIR 22 merges nvvm.cp.async.mbarrier.arrive.shared into
  # nvvm.cp.async.mbarrier.arrive and adds reduction operands to nvvm.barrier;
  # XTDK keeps the split op and legacy signature. Keep this probe independent.
  set(_NVVM_HDR "${LLVM_INCLUDE_DIRS}/mlir/Dialect/LLVMIR/NVVMOps.h.inc")
  if(EXISTS "${_NVVM_HDR}")
    file(READ "${_NVVM_HDR}" _NVVM_SRC)
    if(NOT _NVVM_SRC MATCHES "CpAsyncMBarrierArriveSharedOp")
      message(STATUS "[XPU] NVVM: merged cp.async.mbarrier.arrive (public MLIR)")
      add_compile_definitions(TRITON_MLIR_NVVM_MERGED_MBARRIER_ARRIVE)
    else()
      message(STATUS "[XPU] NVVM: split cp.async.mbarrier.arrive[.shared] (XTDK MLIR)")
    endif()
    # Public MLIR 22 adds reductionOp/reductionPredicate operands to nvvm.barrier.
    if(_NVVM_SRC MATCHES "BarrierReductionAttr")
      message(STATUS "[XPU] NVVM: barrier with reduction operands (public MLIR)")
      add_compile_definitions(TRITON_MLIR_NVVM_BARRIER_REDUCTION)
    endif()
  endif()
endif()
endmacro()


# XPU adjustments for the `triton` Python-module target, called right after it is
# linked against TRITON_LIBRARIES: the conceal-IR definition, the GCC 11 static
# libstdc++ archive, and the repeated public-LLVM archives.
macro(flagtree_configure_backend_triton_target target)
  if(FLAGTREE_BACKEND STREQUAL "xpu")
    target_compile_definitions(${target} PRIVATE TRITON_CONCEAL_IR=1)
  endif()
  # Link the GCC 11 static libstdc++ archive after all LLVM archives.
  if(DEFINED _GCC11_LIBSTDCXX AND _GCC11_LIBSTDCXX)
    target_link_libraries(${target} PRIVATE "${_GCC11_LIBSTDCXX}")
  endif()
  # Public LLVM 22 needs LLVMPlugins for llvm::PassPlugin::Load, while the Arith and
  # MemRef archives appear before their users. Repeat all three after the main link so
  # static linking can pull the required members; repeating archives is safe.
  # Gate on the imported target: LLVM-19-era toolchains (XTDK etc.) do not ship
  # libLLVMPlugins.a, and an unresolved name degrades to `-lLLVMPlugins` (link error).
  if(TARGET LLVMPlugins)
    target_link_libraries(${target} PRIVATE
      LLVMPlugins
      MLIRArithDialect
      MLIRMemRefDialect
    )
  endif()
endmacro()
