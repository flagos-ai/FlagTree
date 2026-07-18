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

set(LLVM_TARGET_DEFINITIONS DistributedOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=distributed)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=distributed)
mlir_tablegen(OpsConversions.inc -gen-llvmir-conversions)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(DistributedDialect DistributedDialect dialects/ -gen-dialect-doc)
add_mlir_doc(DistributedOps DistributedOps dialects/ -gen-op-doc)
add_public_tablegen_target(DistributedTableGen)

set(LLVM_TARGET_DEFINITIONS DistributedAttrDefs.td)
mlir_tablegen(DistributedEnums.h.inc -gen-enum-decls)
mlir_tablegen(DistributedEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(DistributedAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(DistributedAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(DistributedAttrDefsIncGen)
