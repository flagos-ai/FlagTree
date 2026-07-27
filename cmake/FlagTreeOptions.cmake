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

macro(flagtree_configure_options)
  set(FLAGTREE_BACKEND "$ENV{FLAGTREE_BACKEND}")
  set(FLAGTREE_DEFAULT_OPTION ON)
  if(FLAGTREE_BACKEND)
    set(FLAGTREE_DEFAULT_OPTION OFF)
    add_definitions(-DFLAGTREE_BACKEND=\"${FLAGTREE_BACKEND}\")
  endif()

  set(FLAGCX_ENABLED OFF)
  set(FLAGCX_SUPPORT_BACKENDS nvidia)
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
    endif()
    set(FLAGTREE_TLE OFF)
    remove_definitions(-D__TLE__)
    list(REMOVE_ITEM LLVM_TABLEGEN_FLAGS -D__TLE__)
  elseif(FLAGTREE_BACKEND STREQUAL "sunrise")
    find_package(Python3 3.10 REQUIRED COMPONENTS Development.Module Interpreter)
  endif()

  set(FLAGTREE_PLUGIN "$ENV{FLAGTREE_PLUGIN}")
  if(FLAGTREE_PLUGIN)
    add_definitions(-D__FLAGTREE_PLUGIN__)
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
  if(FLAGTREE_BACKEND MATCHES "^(enflame|hcu|rpu|thrive|metax|xpu|tileir)$")
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
     "^(xpu|cambricon|aipu|tsingmicro|enflame|rpu|thrive|tileir)$")
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
    option(TRITON_BUILD_GLUON
      "Build the Gluon IR Python bindings (gluon_ir.cc)" OFF)
    include_directories(${TRITON_CORE_SOURCE_DIR}/include)
    include_directories(${TRITON_CORE_BINARY_DIR}/include)
    include_directories(${TRITON_CORE_SOURCE_DIR}/backend/include)
    include_directories(${TRITON_CORE_BINARY_DIR}/backend/include)
    if(FLAGTREE_ILUVATAR_TLE)
      include_directories(${TRITON_CORE_SOURCE_DIR}/tle/include)
      include_directories(${TRITON_CORE_BINARY_DIR}/tle/include)
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


macro(flagtree_python_src_path_set output_python_src_path default_path)
  set(${output_python_src_path}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND}/python/src)
  if(NOT (FLAGTREE_BACKEND AND EXISTS "${${output_python_src_path}}"))
    set(${output_python_src_path} "${default_path}")
  endif()
endmacro()


macro(flagtree_python_link_libraries)
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${pybind11_INCLUDE_DIR})
  link_directories(${Python3_LIBRARY_DIRS})
  link_libraries(${Python3_LIBRARIES})
  add_link_options(${Python3_LINK_OPTIONS})
endmacro()


macro(flagtree_configure_python_plugins)
  # We always build proton dialect because core Triton conversion libraries link
  # ProtonIR. XPU-specific Proton GPU lowering wrappers are disabled inside the
  # Proton plugin instead of skipping the whole dialect.
  if(FLAGTREE_BACKEND STREQUAL "hcu")
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

  # Add TLE plugin
  if(FLAGTREE_TLE)
    list(APPEND TRITON_PLUGIN_NAMES "tle")
    add_subdirectory(third_party/tle)
    flagtree_add_tle_generated_header_dependencies()
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

  # Native compiler targets include TLE generated headers under __TLE__ guards.
  # The TLE dialect is added after the core libraries, so the dependency must be
  # attached explicitly once the TLE tablegen targets exist; otherwise a clean
  # parallel build can compile those libraries before the generated .inc files.
  foreach(_flagtree_tle_header_target IN ITEMS
      TritonAnalysis
      TritonToTritonGPU
      TritonGPUTransforms
      TritonNvidiaGPUTransforms
      TritonNVIDIAGPUToLLVM
      TritonGPUToLLVM
      NVHopperTransforms
      triton
      triton-opt
      triton-reduce
      triton-lsp
      triton-llvm-opt
      triton-tensor-layout)
    if(TARGET ${_flagtree_tle_header_target})
      add_dependencies(${_flagtree_tle_header_target}
        ${_flagtree_tle_codegen_deps})
    endif()
  endforeach()
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
