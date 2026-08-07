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

#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"


def TritonPPUGPUAccelerateMatmul : Pass<"tritonppugpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "ppu accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., PPU tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonPPUAIULoweringPass : Pass<"tritonppu-aiu-lowering", "mlir::ModuleOp"> {
  let summary = "ppu aiu lowering";

  let description = [{
    Accelerate data movement from global memory to shared memory
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::ppu_gpu::TritonPPUGPUDialect",
                           "mlir::triton::TritonDialect"];
}


def TlePromoteAsyncLoadToAIUPass : Pass<"tle-promote-async-load-to-aiu", "mlir::ModuleOp"> {
  let summary = "Promote TLE async loads to AIU loads for PPU";

  let description = [{
    Finds tt.LoadOp with the tt.load.async attribute and tensor pointer operands,
    and replaces them with tt.AIULoadOp. Must run before rewrite_tensor_pointer.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

#endif
