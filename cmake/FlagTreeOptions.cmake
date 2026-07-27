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
    # metax use mctle to replace tle
    option(BUILD_MCTLE "use maca triton language extensions" ON)
    list(APPEND TRITON_PLUGIN_NAMES "mctle")
    add_definitions(-D__MCTLE__)

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
