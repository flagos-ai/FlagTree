/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToLLVMPPU : Pass<"convert-triton-gpu-to-llvm-ppu", "mlir::ModuleOp"> {
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
                             "mlir::triton::ppu_gpu::TritonPPUGPUDialect",
                             "mlir::triton::ppugpu::PPUGPUDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
    ];
}

def AllocateSharedMemoryPPU : Pass<"allocate-shared-memory-ppu", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation for PPU";

  let description = [{
    See `allocate-shared-memory` for more details.
  }];

  let options = [
      Option<"computeCapability", "compute-capability",
             "int32_t", /*default*/"80",
             "device compute capability">,
  ];
}

def ConvertLibdeviceFuncToPPU : Pass<"convert-libdevice-func-to-ppu", "mlir::ModuleOp"> {
  let summary = "Convert libdevice function calls to __ppu prefixed";
  let description = [{
    This pass walks the module and converts all libdevice function declarations and
    calls to __ppu prefixed.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // TRITONGPU_CONVERSION_PASSES
