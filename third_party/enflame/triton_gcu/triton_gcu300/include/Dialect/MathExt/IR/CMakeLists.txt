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

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS MathExtOps.td)
mlir_tablegen(MathExtOps.h.inc -gen-op-decls)
mlir_tablegen(MathExtOps.cpp.inc -gen-op-defs)
# mlir_tablegen(MathExtOpsEnums.h.inc -gen-enum-decls)
# mlir_tablegen(MathExtOpsEnums.cpp.inc -gen-enum-defs)
# mlir_tablegen(MathExtOpsAttributes.h.inc -gen-attrdef-decls)
# mlir_tablegen(MathExtOpsAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_doc(MathExtOps MathExtOps_gcu300 dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MathExtDialect.td)
mlir_tablegen(MathExtDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MathExtDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(MathExtDialect MathExtDialect_gcu300 dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MathExtTypes.td)
mlir_tablegen(MathExtTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MathExtTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(MathExtTableGen_gcu300)
