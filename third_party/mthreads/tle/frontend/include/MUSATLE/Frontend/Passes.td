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
