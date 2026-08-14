#ifndef ILUVATAR_TLE_PASSES
#define ILUVATAR_TLE_PASSES

include "mlir/Pass/PassBase.td"

def TritonIluvatarTleEarlyAssignMemorySpace
    : Pass<"triton-iluvatar-tle-early-assign-memory-space", "mlir::ModuleOp"> {
  let summary = "materialize TLE tt.memory_space annotations for iluvatar";
  let description = [{
    Rewrite tensors marked with `tt.memory_space = "shared_memory"` (produced by
    `tle.gpu.memory_space`) into explicit shared-memory memdesc traffic before
    iluvatar layout/MMA transforms run. Loads unconditionally go through the
    async global-to-shared copy path; every other producer is materialized via
    an initialized `local_alloc` plus `local_load`.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleOptimizeLocalPointerAsyncStores
    : Pass<"triton-iluvatar-tle-optimize-local-pointer-async-stores", "mlir::ModuleOp"> {
  let summary = "fuse tle.copy global load + local_pointers store into async copy";
  let description = [{
    Recognize the `tt.load` (global) followed by `tt.store` into an
    `iluvatar_tle.local_pointers` (sub)view staging that `tle.gpu.copy`
    (GM_TO_LOCAL) emits, and rewrite it into
    `ttg.async_copy_global_to_local` plus `ttg.async_commit_group` /
    `ttg.async_wait`. Consecutive non-overlapping copies share a single
    commit/wait. Runs before the local-pointer store/load lowering so the
    load->store pattern is still visible; patterns that do not match keep their
    ordinary synchronous semantics.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleInsertLocalPointerBarriers
    : Pass<"triton-iluvatar-tle-insert-local-pointer-barriers", "mlir::ModuleOp"> {
  let summary = "insert CTA barriers between iluvatar_tle.local_pointers stores and loads";
  let dependentDialects = ["mlir::gpu::GPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleOptimizeLocalPointerLoads
    : Pass<"triton-iluvatar-tle-optimize-local-pointer-loads", "mlir::ModuleOp"> {
  let summary = "rewrite full-view iluvatar_tle.local_pointers loads into ttg.local_load";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleOptimizeLocalPointerStores
    : Pass<"triton-iluvatar-tle-optimize-local-pointer-stores", "mlir::ModuleOp"> {
  let summary = "rewrite iluvatar_tle.local_pointers stores into ttg.local_store";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleLowerAsyncLoad
    : Pass<"triton-iluvatar-tle-lower-async-load", "mlir::ModuleOp"> {
  let summary = "lower TLE async load hints to iluvatar async copies";
  let description = [{
    Rewrite `tt.load` operations marked with `tt.load.async = true` (produced by
    `tle.load(..., is_async=True)`) into `ttg.async_copy_global_to_local` plus
    commit/wait/local_load when the load can be represented by the iluvatar
    async global-to-shared copy path. Unsupported hints are dropped so the load
    keeps ordinary synchronous `tt.load` correctness semantics.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleOptimizeExclusiveCumsumLayouts
    : Pass<"triton-iluvatar-tle-optimize-exclusive-cumsum-layouts", "mlir::ModuleOp"> {
  let summary = "fold convert_layout around iluvatar_tle.exclusive_cumsum";
  let description = [{
    Fold patterns of the form
      ttg.convert_layout -> iluvatar_tle.exclusive_cumsum -> ttg.convert_layout
    when the source and sink layouts are identical, so the cumsum runs directly
    on the canonical layout and the redundant converts are removed. Runs before
    `triton-iluvatar-tle-lower-exclusive-cumsum` so the op is still present.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleLowerExclusiveCumsum
    : Pass<"triton-iluvatar-tle-lower-exclusive-cumsum", "mlir::ModuleOp"> {
  let summary = "lower tle.exclusive_cumsum into tt.scan + tt.reduce";
  let description = [{
    Rewrite `iluvatar_tle.exclusive_cumsum` (produced by `tle.cumsum`) into an
    inclusive `tt.scan` (add combiner) minus the source for the exclusive
    result, plus an index-select `tt.reduce` to recover the total sum. Runs
    after TritonToTritonGPU conversion so the emitted scan/reduce carry the
    surrounding TritonGPU encodings.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleLowerPipeToBarriers
    : Pass<"triton-iluvatar-tle-lower-pipe-to-barriers", "mlir::ModuleOp"> {
  let summary = "lower CTA SMEM TLE pipe ops to software mbarrier ops";
  let description = [{
    Lower `iluvatar_tle.pipe.*` lifecycle ops to `ttng.init/wait/arrive_barrier`
    plus an optional close-tag SMEM ring. Intended for ivcore11 where hardware
    mbarrier is unavailable and BarrierOpToLLVM emulates the protocol with
    shared-memory atomics. MVP covers CTA-scoped SPSC/SPMC local-store pipes
    used with `ttg.warp_specialize`.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

#endif // ILUVATAR_TLE_PASSES
