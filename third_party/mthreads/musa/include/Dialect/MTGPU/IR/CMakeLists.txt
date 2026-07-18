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

set(LLVM_TARGET_DEFINITIONS MTGPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=mtgpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=mtgpu)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(MTGPUDialect MTGPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(MTGPUOps MTGPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(MTGPUTableGen)

set(LLVM_TARGET_DEFINITIONS MTGPUTypes.td)
mlir_tablegen(MTGPUTypes.h.inc -gen-typedef-decls -typedefs-dialect=mtgpu)
mlir_tablegen(MTGPUTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=mtgpu)
add_public_tablegen_target(MTGPUTypesIncGen)
