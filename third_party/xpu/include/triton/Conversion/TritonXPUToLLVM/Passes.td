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

#ifndef TTX2LLVM_CONVERSION_PASSES
#define TTX2LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonXPUToLLVM: Pass<"convert-triton-xpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonXPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonXPUToLLVMPass()";

    let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

   let options = [
        Option<"xpu_arch", "xpu_arch",
               "uint32_t", /*default*/"3",
               "XPU Architecture">,
        Option<"buffer_size", "buffer_size",
               "uint32_t", /*default*/"128",
               "Buffer Size of LM">,
        Option<"isUseMaskZero", "is-use-mask-zero",
               "bool", /*default*/"0",
               "Whether Use Mask Zero Method">,
   ];
}

def AllocateXPUSharedMemory: Pass<"allocate-xpu-shared-memory", "mlir::ModuleOp"> {
    let summary = "Allocate shared memory for TritonXPU ops";
    let description = [{
      Attach allocation.offset and ttg.shared using XPU-specific scratch sizes.
    }];
    let constructor = "mlir::triton::createAllocateXPUSharedMemoryPass()";
    let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}


#endif // TTX2LLVM_CONVERSION_PASSES
