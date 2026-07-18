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

# flagtree tle
# IR headers for TLE dialect
# Add IR header files here

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TleOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TleOps TleOps dialects/ -gen-op-doc)

if (FLAGCX_ENABLED)
set(LLVM_TARGET_DEFINITIONS FlagCxOps.td)
mlir_tablegen(FlagCxOps.h.inc -gen-op-decls)
mlir_tablegen(FlagCxOps.cpp.inc -gen-op-defs)
add_mlir_doc(FlagCxOps FlagCxOps dialects/ -gen-op-doc)

endif()

set(LLVM_TARGET_DEFINITIONS TleDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tle)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tle)
add_mlir_doc(TleDialect TleDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TleAttrDefs.td)
mlir_tablegen(TleAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TleAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TleTableGen)
