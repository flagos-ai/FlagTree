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

set(LLVM_TARGET_DEFINITIONS Passes.td)
if(FLAGTREE_MTHREADS_TLE)
  set(_TLE_TABLEGEN_DEFS -D__TLE__)
  set(_TLE_TABLEGEN_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../tle/dialect/include/MUSATLE/Transforms/Passes.td)
else()
  set(_TLE_TABLEGEN_DEFS "")
  set(_TLE_TABLEGEN_DEPENDS "")
endif()

set(LLVM_TARGET_DEPENDS ${_TLE_TABLEGEN_DEPENDS})
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonMUSAGPU
              ${_TLE_TABLEGEN_DEFS})
add_public_tablegen_target(TritonMUSAGPUTransformsIncGen)
