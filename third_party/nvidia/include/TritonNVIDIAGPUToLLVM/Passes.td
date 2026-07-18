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

#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToLLVM : Pass<"convert-triton-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::triton::nvgpu::NVGPUDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}
def AllocateSharedMemoryNv : Pass<"allocate-shared-memory-nv", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation for Nvidia";

  let description = [{
    See `allocate-shared-memory` for more details.
  }];

  let options = [
      Option<"computeCapability", "compute-capability",
             "int32_t", /*default*/"80",
             "device compute capability">,
      Option<"ptxVersion", "ptx-version",
             "int32_t", /*default*/"80",
             "PTX version">,
  ];
}


def ConvertWarpSpecializeToLLVM : Pass<"convert-warp-specialize-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower `ttg.warp_specialize` to LLVM";
  let description = [{
    The `convert-warp-specialize-to-llvm` pass performs codegen for warp
    specialization. It is a function-level transformation that rewrites
    warp-specialized kernels by using shared memory and barriers to communicate
    states between the default warpgroup and the worker warps.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::NVVM::NVVMDialect"];
}

#endif // TRITONGPU_CONVERSION_PASSES
