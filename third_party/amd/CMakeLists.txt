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
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  add_triton_plugin(TritonAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM)
  target_link_libraries(TritonAMD PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(test)
