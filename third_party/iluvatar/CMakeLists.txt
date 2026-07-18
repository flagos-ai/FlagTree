# Copyright 2026 FlagOS Contributors
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/backend/include)
if(FLAGTREE_ILUVATAR_TLE)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/include)
  add_subdirectory(tle)
endif()
add_subdirectory(backend/include)
add_subdirectory(backend/lib)
if(TRITON_BUILD_PYTHON_MODULE)
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  if(FLAGTREE_ILUVATAR_TLE)
    set(_ILUVATAR_TLE_PLUGIN_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/tle/triton_iluvatar_tle.cc)
    set(_ILUVATAR_TLE_PLUGIN_LIBS IluvatarTleIR IluvatarTleTransforms)
    set(_ILUVATAR_TLE_PLUGIN_DEPS IluvatarTleTableGen IluvatarTleTransformsIncGen)
  else()
    set(_ILUVATAR_TLE_PLUGIN_SOURCES "")
    set(_ILUVATAR_TLE_PLUGIN_LIBS "")
    set(_ILUVATAR_TLE_PLUGIN_DEPS "")
  endif()
  add_triton_plugin(TritonILUVATAR
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_iluvatar.cc
    ${_ILUVATAR_TLE_PLUGIN_SOURCES}
    LINK_LIBS TritonILUVATARGPUToLLVM ${_ILUVATAR_TLE_PLUGIN_LIBS})
  if(FLAGTREE_ILUVATAR_TLE)
    add_dependencies(TritonILUVATAR ${_ILUVATAR_TLE_PLUGIN_DEPS})
  endif()
  target_link_libraries(TritonILUVATAR PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
# add_subdirectory(test)
