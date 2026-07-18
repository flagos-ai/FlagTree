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

set(LLVM_TARGET_DEFINITIONS TritonAMDGPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=amdg)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=amdg)
mlir_tablegen(OpsConversions.inc -gen-llvmir-conversions)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TritonAMDGPUDialect TritonAMDGPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TritonAMDGPUOps TritonAMDGPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(TritonAMDGPUTableGen)

set(LLVM_TARGET_DEFINITIONS TritonAMDGPUAttrDefs.td)
mlir_tablegen(TritonAMDGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonAMDGPUEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TritonAMDGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonAMDGPUAttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS TritonAMDGPUOpInterfaces.td)
mlir_tablegen(TritonAMDGPUOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TritonAMDGPUOpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TritonAMDGPUAttrDefsIncGen)
