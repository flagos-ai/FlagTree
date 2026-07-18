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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  set(PROTON_PLUGIN_LINK_LIBS
    ProtonToProtonGPU
    ProtonGPUToLLVM
    ProtonNVIDIAGPUToLLVM
    ProtonAnalysis
  )
  if(TRITON_ENABLE_AMD)
    list(APPEND PROTON_PLUGIN_LINK_LIBS ProtonAMDGPUToLLVM)
  endif()
  add_triton_plugin(TritonProton ${CMAKE_CURRENT_SOURCE_DIR}/triton_proton.cc LINK_LIBS ${PROTON_PLUGIN_LINK_LIBS})
  target_link_libraries(TritonProton PRIVATE Python3::Module pybind11::headers)
endif()
