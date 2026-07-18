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

#ifndef MTHREADS_MUSATLE_FRONTEND_PASSES
#define MTHREADS_MUSATLE_FRONTEND_PASSES

include "mlir/Pass/PassBase.td"

def TritonMUSAGPUTLEEarlyAssignMemorySpace
    : Pass<"tritonmusa-tle-early-assign-memory-space", "mlir::ModuleOp"> {
  let summary = "Materialize TLE memory-space annotations for MUSA";
  let description = [{
    Rewrite tensors marked with `tt.memory_space = "shared_memory"` into
    explicit shared-memory memdesc traffic before MUSA-specific layout and MMA
    transforms run. Legal loads are lowered through MUSA async copy plus
    commit/wait; all other producers use initialized local_alloc/local_load
    materialization to preserve tensor semantics.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonMUSAGPUTLELowerAsyncLoad
    : Pass<"tritonmusa-tle-lower-async-load", "mlir::ModuleOp"> {
  let summary = "Lower TLE async load hints to MUSA async copies";
  let description = [{
    Rewrite `tt.load` operations marked with `tt.load.async = true` into
    `ttg.async_copy_global_to_local` plus commit/wait/local_load when the
    load can be represented by the MUSA async global-to-shared copy path.
    Unsupported hints are dropped so `tle.load(..., is_async=True)` remains
    a scheduling hint with ordinary `tt.load` correctness semantics.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // MTHREADS_MUSATLE_FRONTEND_PASSES
