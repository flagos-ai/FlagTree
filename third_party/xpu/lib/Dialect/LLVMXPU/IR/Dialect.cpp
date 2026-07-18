// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "triton/Dialect/LLVMXPU/IR/Dialect.h" // before cpp.inc

#include "triton/Dialect/LLVMXPU/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Initialization
//===----------------------------------------------------------------------===//

void ::mlir::LLVM::XPU::LLVMXPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST // declare
#include "triton/Dialect/LLVMXPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES // define
#include "triton/Dialect/LLVMXPU/IR/Ops.cpp.inc"

mlir::LogicalResult
mlir::LLVM::XPU::LLVMXPUDialect::verifyOperationAttribute(Operation *op,
                                                          NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (attr.getName() == LLVMXPUDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << LLVMXPUDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}
