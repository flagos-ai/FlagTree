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

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TRITON_CUDA_TILE_TEST_DEPENDS
  triton-cuda-tile-opt
  triton-opt
  triton-llvm-opt
)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")

# This target can be used to just build the dependencies
# for the check-triton-cuda-tile target without executing the tests.
# This is useful for bots when splitting the build step from
# the testing step.
add_custom_target(check-triton-cuda-tile-build-only
  DEPENDS ${TRITON_CUDA_TILE_TEST_DEPENDS}
)

add_lit_testsuite(check-triton-cuda-tile "Running the triton-cuda-tile tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TRITON_CUDA_TILE_TEST_DEPENDS}
  )
set_target_properties(check-triton-cuda-tile PROPERTIES FOLDER "Tests")

add_lit_testsuites(triton-cuda-tile ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TRITON_CUDA_TILE_TEST_DEPENDS}
)
