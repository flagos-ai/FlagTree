/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
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

#ifndef TRITONCOMMONGPU_CONVERSION_PASSES
#define TRITONCOMMONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateSharedMemory : Pass<"allocate-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

def TritonGPUGlobalScratchAllocationPass : Pass<"tritongpu-global-scratch-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Assign global scratch memory allocation";

  let description = [{
    Decide on global scratch space memory allocation and assign attributes to each allocation.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonGPUAllocateWarpGroups : Pass<"tritongpu-allocate-warp-groups", "mlir::ModuleOp"> {
  let summary = "Allocate warp groups";

  let description = [{
    The `tritongpu-allocate-warp-groups` pass performs warpgroup allocation for
    a GPU program. When a GPU program contains warp specialization, additional
    warps are launched in addition to the "default" warp group. The "default"
    warpgroup executes top-level code in a `tt.func` and its size is specified
    by the user via the `num_warps` argument.

    This pass analyzes `ttg.warp_specialize` ops in the program and determines
    the total number of needed warps, then attaches the range of warp IDs to
    each warpgroup function.
  }];
}

#endif
