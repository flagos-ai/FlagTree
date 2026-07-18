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

#ifndef TRITONMUSAGPU_CONVERSION_PASSES
#define TRITONMUSAGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonMUSAGPUToLLVM : Pass<"convert-triton-musagpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM for MUSA";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::musa::MUSADialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"31",
               "device compute capability">,
    ];
}

def AllocateMUSASharedMemory : Pass<"allocate-musa-shared-memory", "mlir::ModuleOp"> {
    let summary = "Allocate shared memory for MUSA";

    let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"31",
               "device compute capability">,
    ];
}

#endif
