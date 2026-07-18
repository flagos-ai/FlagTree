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

set(LLVM_TARGET_DEFINITIONS TritonGCUDialect.td)
mlir_tablegen(TritonGCUDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TritonGCUDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TritonGCUTypes.td)
mlir_tablegen(TritonGCUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TritonGCUTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TritonGCUOps.td)
mlir_tablegen(TritonGCUOps.h.inc -gen-op-decls)
mlir_tablegen(TritonGCUOps.cpp.inc -gen-op-defs)
mlir_tablegen(TritonGCUOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonGCUOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TritonGCUOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonGCUOpsAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS TritonGCUInterfaces.td)
mlir_tablegen(TritonGCUAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TritonGCUAttrInterfaces.cpp.inc -gen-attr-interface-defs)

add_public_tablegen_target(TritonGCUTableGen_${arch})
