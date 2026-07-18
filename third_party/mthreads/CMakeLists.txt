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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/frontend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/frontend/include)
if(FLAGTREE_MTHREADS_TLE)
  add_subdirectory(tle)
endif()
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(musa)
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_MTHREADS_TLE)
    set(_MTHREADS_TLE_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/triton_mthreads_tle.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tle/frontend/triton_mthreads_frontend.cc)
    set(_MTHREADS_TLE_PLUGIN_LIBS
        MUSATLEIR MUSATLETransforms MUSATLEFrontendTransforms)
    set(_MTHREADS_TLE_PLUGIN_DEPS
        MUSATLETableGen
        MUSATLETransforms
        MUSATLEFrontendTransformsIncGen
        MUSATLEFrontendTransforms)
  else()
    set(_MTHREADS_TLE_PLUGIN_SOURCES "")
    set(_MTHREADS_TLE_PLUGIN_LIBS "")
    set(_MTHREADS_TLE_PLUGIN_DEPS "")
  endif()
  add_triton_plugin(TritonMthreads ${CMAKE_CURRENT_SOURCE_DIR}/triton_mthreads.cc
                              ${_MTHREADS_TLE_PLUGIN_SOURCES}
                    LINK_LIBS TritonMUSAGPUToLLVM MTGPUToLLVM
                              TritonMUSAGPUTransforms
                              ${_MTHREADS_TLE_PLUGIN_LIBS}
                              MLIRMTVMToLLVMIRTranslation)
  add_dependencies(TritonMthreads
                   MUSATableGen
                   MUSAAttrDefsIncGen
                   ${_MTHREADS_TLE_PLUGIN_DEPS}
                   MTGPUTableGen
                   MTGPUTypesIncGen
                   MTGPUConversionPassIncGen
                   TritonMUSAGPUConversionPassIncGen
                   TritonMUSAGPUTransformsIncGen)
  target_link_libraries(TritonMthreads PRIVATE Python3::Module pybind11::headers)
endif()
add_subdirectory(bin)
