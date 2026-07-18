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

# Compile NVIDIA compat sources for use by TritonInstrumentToLLVM
# (PTXBuilder, createElectPredicateWarp0, etc.)
set(NVIDIA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../nvidia/lib/TritonNVIDIAGPUToLLVM)

add_library(TritonNVIDIACompat STATIC
  ${NVIDIA_SRC_DIR}/PTXAsmFormat.cpp
  ${NVIDIA_SRC_DIR}/Utility.cpp
)
target_include_directories(TritonNVIDIACompat PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../nvidia/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)
