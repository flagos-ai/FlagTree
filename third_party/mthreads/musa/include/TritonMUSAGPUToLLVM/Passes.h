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

#ifndef TRITONMUSAGPU_CONVERSION_TRITONMUSAGPUTOLLVM_PASSES_H
#define TRITONMUSAGPU_CONVERSION_TRITONMUSAGPUTOLLVM_PASSES_H

#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "musa/include/TritonMUSAGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonMUSAGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMUSAGPUToLLVMPass(int32_t computeCapability);
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateMUSASharedMemoryPass(int32_t computeCapability);

#define GEN_PASS_REGISTRATION
#include "musa/include/TritonMUSAGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
