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

#ifndef PROTONNVIDIAGPU_TO_LLVM_PASSES
#define PROTONNVIDIAGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonNvidiaGPUToLLVM : Pass<"convert-proton-nvidia-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let description = [{
        Convert ProtonGPU to LLVM using Nvidia-specific lowering patterns.
    }];
    let constructor = "mlir::triton::proton::gpu::createConvertProtonNvidiaGPUToLLVMPass(80, 80)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::proton::ProtonDialect",
                             "mlir::triton::proton::gpu::ProtonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}

#endif // PROTONNVIDIAGPU_TO_LLVM_PASSES
