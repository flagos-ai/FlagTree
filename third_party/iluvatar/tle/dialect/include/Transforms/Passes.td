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

    A copy whose destination is a TCU shared tile and whose source is an
    SME-eligible kernel argument is emitted as an SME global->shared transfer
    instead, by recovering the row stride from the pointer arithmetic and
    tagging the pointer with an `isSme` blocked encoding.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
  let options = [
    Option<"useSme", "use-sme", "unsigned", /*default=*/"0",
           "Per-kernel-argument SME eligibility bitmask; bit N marks argument N "
           "as an SME-capable tensor. Computed at launch time by get_corex_sme, "
           "so it also carries the 64-byte row-alignment guarantee.">,
    Option<"numStages", "num-stages", "int64_t", /*default=*/"0",
           "Kernel-wide pipeline depth. A whole-buffer staging that only feeds "
           "a dot inside a loop of this depth is left unfused for "
           "promote-local-store-staging and the software pipeliner.">
  ];
}

def TritonIluvatarTlePromoteLocalStoreStaging
    : Pass<"triton-iluvatar-tle-promote-local-store-staging", "mlir::ModuleOp"> {
  let summary =
      "promote transient TLE local_store staging to pipelineable local_alloc";

  let description = [{
    This pass rewrites safe loop-local staging patterns of the form:

      %v = tt.load ...
      ttg.local_store %v, %smem
      ... users of %smem ...

    into:

      %stage = ttg.local_alloc %v
      ... users of %stage ...

    for buffers that are overwritten every loop iteration and only read after
    the store. The rewrite exposes the standard TritonGPU load -> local_alloc
    pattern so the existing software pipeliner can expand it into double
    buffered async copies when the loop has ``tt.num_stages > 1``.

    Mirrors `triton-tle-promote-local-store-staging`. The iluvatar deviation is
    that the pipeline depth may come from the kernel-wide option below when the
    loop carries no `tt.num_stages` hint.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];

  let options = [
    Option<"numStages", "num-stages", "int64_t", /*default=*/"0",
           "Kernel-wide pipeline depth, used when the loop carries no "
           "tt.num_stages hint of its own.">
  ];
}

def TritonIluvatarTleMarkSmeDotOperands
    : Pass<"triton-iluvatar-tle-mark-sme-dot-operands", "mlir::ModuleOp"> {
  let summary = "flag dot operands read back from an SME global->shared copy";
  let description = [{
    An int8 row-major SME copy writes the `rowxfb8` shared layout, which is only
    GF(2)-linear up to a bit-7 offset correction that LocalLoadOpConversion
    applies when `isIluvatarRowXfb8LocalLoad` holds. For a dot operand that test
    additionally requires a non-zero `useSme`, which `AccelerateMatmul` only sets
    for loads it recognizes itself and therefore never for a copy fused out of
    `tle.gpu.copy`. Recover it by tracing the `ttg.local_load` source memdesc
    back to the SME copy that wrote it and retyping the result in place. Runs
    once layouts are final, so no later pass can drop the flag.

    A tile staged by `triton-iluvatar-tle-promote-local-store-staging` reaches
    the dot as `local_alloc(load) -> local_load`, which
    `hasSmeCompatibleLoadChain` also rejects, leaving `useSme` at 0 so
    `matmul-smeload` declines the load entirely. Recover the flag for that shape
    too, applying the same eligibility rules the async-store fusion uses. That
    run has to precede `matmul-smeload`.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
  let options = [
    Option<"useSme", "use-sme", "unsigned", /*default=*/"0",
           "Per-kernel-argument SME eligibility bitmask, as passed to the "
           "async-store fusion. Only consulted for the staged local_alloc "
           "shape, where no earlier pass established SME eligibility.">
  ];
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

def TritonIluvatarTleConvertArgToMemDesc
    : Pass<"triton-iluvatar-tle-convert-arg-to-memdesc", "mlir::ModuleOp"> {}

def TritonIluvatarTleRemoveRedundantCopy
    : Pass<"triton-iluvatar-tle-remove-redundant-copy", "mlir::ModuleOp"> {}

def TritonIluvatarTleDSLRegionInline
    : Pass<"triton-iluvatar-tle-dslregion-inline", "mlir::ModuleOp"> {}

def TritonIluvatarTleAddDistributedParams
    : Pass<"triton-iluvatar-tle-add-distributed-params", "mlir::ModuleOp"> {

  let summary = "Add distributed runtime parameters to Triton kernel functions";

  let description = [{
    This pass updates Triton kernel function signatures by inserting two
    runtime parameters at the beginning of each kernel function:

      * device_comm_ptr
      * device_mem_ptr

    These parameters provide device-side communication and runtime state
    required by distributed TLE operations.

    All call sites and function argument uses are updated accordingly.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

#endif // ILUVATAR_TLE_PASSES
