// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TTX2LLVM_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H
#define TTX2LLVM_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"

namespace xpu {

// TODO[dyq]: can be used ?
// std::unique_ptr<OperationPass<ModuleOp>>
// createDecomposeUnsupportedConversionsPass(uint32_t xpu_arch);

} // namespace xpu

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonXPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonXPUToLLVMPass(uint32_t xpu_arch, uint32_t buffer_size,
                                 bool isUseMaskZero);
std::unique_ptr<OperationPass<ModuleOp>> createAllocateXPUSharedMemoryPass();

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TTX2LLVM_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H
