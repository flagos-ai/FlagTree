// flagtree tle
#ifndef TRITON_TLE_PASSES
#define TRITON_TLE_PASSES

include "mlir/Pass/PassBase.td"

def TritonTleEarlyAssignMemorySpace
    : Pass<"triton-tle-early-assign-memory-space", "mlir::ModuleOp"> {
  let summary = "early assign memory space";

  let description = [{
    This pass assigns memory spaces to tensor pointers according to `tt.memory_space` attribute on tensors.
    This is done early to enable subsequent passes to make better optimization decisions based on memory spaces.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleSelectEncodings
    : Pass<"triton-tle-select-encodings", "mlir::ModuleOp"> {
  let summary = "select shared encodings for local pointer and related users";

  let description = [{
    This pass performs shared-memory pointer canonicalization and encoding
    selection for `tle.local_pointers` and dependent users so downstream
    conversion sees stable, cost-aware layouts.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleInsertLocalPointerBarriers
    : Pass<"triton-tle-insert-local-pointer-barriers", "mlir::ModuleOp"> {
  let summary =
      "insert CTA barriers between tle.local_pointers stores and loads";

  let description = [{
    This pass ensures data staged through ``tle.local_pointers`` backed shared
    memory buffers is globally visible before the program re-reads the same
    buffer. It looks for sequences of ``tl.store`` operations targeting
    ``tle.local_pointers`` tensors followed by ``tl.load`` reads and inserts a
    ``gpu.barrier`` between the two when needed.
  }];

  let dependentDialects = ["mlir::gpu::GPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleOptimizeLocalPointerLoads
    : Pass<"triton-tle-optimize-local-pointer-loads", "mlir::ModuleOp"> {
  let summary = "rewrite full-view tle.local_pointers loads into ttg.local_load";

  let description = [{
    This pass matches ``tl.load`` operations whose pointer operand is produced
    by ``tle.local_pointers`` with full-view indexing (including zero-indices
    form) and rewrites them into ``ttg.local_load`` directly from the backing
    memdesc. The transform is conservative and only fires when the load has no
    mask/other operands.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleOptimizeLocalPointerStores
    : Pass<"triton-tle-optimize-local-pointer-stores", "mlir::ModuleOp"> {
  let summary = "rewrite tle.local_pointers stores into ttg.local_store";

  let description = [{
    This pass rewrites ``tl.store`` operations that target pointers produced by
    ``tle.local_pointers`` into ``ttg.local_store`` to avoid pointer-based
    shared-memory stores in the backend.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleOptimizeLocalPointerAsyncStores
    : Pass<"triton-tle-optimize-local-pointer-async-stores", "mlir::ModuleOp"> {
  let summary =
      "rewrite gmem load -> tle.local_pointers subview store into async_copy";

  let description = [{
    This pass matches conservative staging patterns of the form:

      %v = tt.load %gptr ...
      tt.store %local_ptr_subview, %v ...

    where ``%local_ptr_subview`` is produced by ``tle.local_pointers`` and
    denotes a static full subview of a backing shared memdesc. The pass
    rewrites the pair into:

      %sub = ttg.memdesc_subslice %base[...]
      %tok = ttg.async_copy_global_to_local %gptr, %sub ...
      %grp = ttg.async_commit_group %tok
      ttg.async_wait %grp {num = 0}

    The transform is conservative and only fires when the store value is
    produced by a single-use global load and the subview offsets are static.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTlePromoteLocalStoreStaging
    : Pass<"triton-tle-promote-local-store-staging", "mlir::ModuleOp"> {
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
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleTileStylePipelineSchedule
    : Pass<"triton-tle-tile-style-pipeline-schedule", "mlir::ModuleOp"> {
  let summary =
      "rewrite eligible Triton schedules into a TileLang-style preload-2 pipeline";

  let description = [{
    This pass runs after Triton loop scheduling and before software pipeline
    expansion. For carefully selected loops, it rewrites the scheduled stages
    from a standard ``stage0(load) + stage1(consume)`` layout into a
    TileLang-style layout with an empty lead stage:

      stage0(load) + stage2(consume)

    together with ``tt.scheduled_max_stage = 2`` and a TLE marker attribute.
    The resulting pipeline shape creates a two-iteration prologue while keeping
    the original Triton software pipeliner as the fallback path when the loop
    is not eligible.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleMaterializeTileStylePipeline
    : Pass<"triton-tle-materialize-tile-style-pipeline", "mlir::ModuleOp"> {
  let summary =
      "materialize eligible TLE tile-style loops into explicit prologue/steady/epilogue form";

  let description = [{
    This pass is the TLE-owned entry point for replacing the native Triton
    software pipeliner on selected Hopper tile-style loops. The intended final
    form is an explicitly materialized pipeline:

      prologue(issue 0, issue 1)
      steady(consume i, issue i + 2)
      epilogue(drain)

    Initial integration keeps the pass disabled by default and preserves
    fallback to the native Triton software pipeliner when the loop is not
    handled.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleDowngradeInvalidAsyncCopy
    : Pass<"triton-tle-downgrade-invalid-async-copy", "mlir::ModuleOp"> {
  let summary =
      "legalize required TLE cp.async copies and downgrade optional ones";

  let description = [{
    This pass is TLE-owned because it repairs async copies created or exposed
    by TLE staging/pipeline transforms. For explicit required copies, it first
    redistributes a provably contiguous blocked source layout so each active
    thread owns a legal 4/8/16-byte transaction while preserving the async op
    and token chain. A required copy fails when no such layout can be proven.
    Other invalid async copies are replaced with a synchronous ``tt.load`` plus
    ``ttg.local_store`` before software pipelining.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleOptimizeExclusiveCumsumLayouts
    : Pass<"triton-tle-optimize-exclusive-cumsum-layouts", "mlir::ModuleOp"> {
  let summary = "fold convert_layout around tle.exclusive_cumsum";

  let description = [{
    This pass folds patterns of the form:
      ttg.convert_layout -> tle.exclusive_cumsum -> ttg.convert_layout
    when the source and sink layouts can be made identical, so the cumsum
    runs directly on the canonical layout and redundant converts are removed.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleLowerExclusiveCumsum
    : Pass<"triton-tle-lower-exclusive-cumsum", "mlir::ModuleOp"> {
  let summary = "lower tle.exclusive_cumsum with a specialized Triton IR pattern";

  let description = [{
    This pass lowers ``tle.exclusive_cumsum`` into Triton IR with a
    cumsum-specialized pattern: one ``tt.scan`` for prefix accumulation and an
    index-select ``tt.reduce`` to extract the tail value used as total sum.
    This avoids a second add-reduction over the original input.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleLowerAsyncLoad
    : Pass<"triton-tle-lower-async-load", "mlir::ModuleOp"> {
  let summary = "Lower TLE async load operations";

  let description = [{
    This pass lowers tl::LoadOp with tt.load.async to target-specific GPU operations by converting
    high-level asynchronous load operations into appropriate TritonGPU primitives.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleLowerTmaCopy
    : Pass</*cli-arg*/ "triton-tle-lower-tma-copy", /*Op*/ "mlir::ModuleOp"> {

  let summary = "Lower TLE TMA copy operations";

  let description = [{
    This pass lowers tle::TmaCopyOp to target-specific GPU operations by converting
    high-level TMA (Tensor Memory Accelerator) copy operations into appropriate
    TritonGPU primitives. It handles the transformation of abstract copy operations
    into concrete GPU memory transfer instructions.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::tle::TleDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonTleScheduleTmaStoreSync
    : Pass<"triton-tle-schedule-tma-store-sync", "mlir::ModuleOp"> {
  let summary = "schedule explicit TLE TMA store commit and wait operations";

  let description = [{
    This pass owns synchronization for TLE shared-to-global TMA stores lowered
    by ``triton-tle-lower-tma-copy``. It groups adjacent TLE TMA stores behind a
    single explicit ``tle.tma_store.commit_group`` and sinks
    ``ttng.async_tma_store_wait`` to the first operation that can reuse or
    release the source shared-memory buffer. Native Triton TMA stores keep their
    standard implicit commit behavior.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::nvws::NVWSDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleCoalesceBarrierInitialization
    : Pass<"triton-tle-coalesce-barrier-initialization", "mlir::ModuleOp"> {
  let summary = "coalesce static barrier initialization across CTA workers";

  let description = [{
    This post-allocation pass replaces a sufficiently large set of top-level
    ``ttng.init_barrier`` operations with one ``tle.init_barrier_group``. The
    grouped op records compiler-assigned shared-memory offsets and lets the
    target lowering initialize independent barriers in parallel across the
    default worker threads. Small kernels retain the native scalar lowering.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];

  let options = [
    Option<"minBarrierCount", "min-barrier-count",
           "int32_t", /*default*/"128",
           "minimum number of static barriers required for grouping">
  ];
}

def TritonTleLowerPipeToNvws
    : Pass<"triton-tle-lower-pipe-to-nvws", "mlir::ModuleOp"> {
  let summary = "lower CTA SMEM TLE pipe operations to NVWS token operations";

  let description = [{
    This pass lowers the strict CTA-scoped SMEM SPSC pipe IR produced by
    tle.pipe to NVWS token operations. Close-aware waits are backed by a
    per-pipe SMEM close-tag ring, separate from the NVWS full/empty token.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::nvws::NVWSDialect"];
}

def TritonTleRestorePipeFunctionCalls
    : Pass<"triton-tle-restore-pipe-function-calls", "mlir::ModuleOp"> {
  let summary = "restore noinline pipe helper calls after NVWS state materialization";

  let description = [{
    Pipe analysis temporarily inlines private helpers so each pipe lifecycle
    operation can be associated with its concrete NVWS control state. After
    NVWS token lowering has materialized barriers and task metadata, this pass
    outlines the marked call scopes again. Repeated calls to one specialized
    helper share one private function and pass the concrete control state as
    ordinary TTGIR operands.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def TritonTleLowerTaskGrid
    : Pass<"triton-tle-lower-task-grid", "mlir::ModuleOp"> {
  let summary = "validate and erase TLE task-grid dependency annotations";

  let description = [{
    This MVP pass owns the dialect-level contract for ``tle.task_grid``. It
    checks that ``tile_id`` and ``commit`` markers reference a preceding
    matching ``tle.task_grid.create`` and that field and shape metadata stays
    consistent with the declared grid.

    Automatic device-scope barrier/atomic generation is intentionally left for
    scheduler codegen in a later pass revision. After validation, unused marker
    operations are erased so the existing TritonGPU lowering pipeline does not
    need to legalize them.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonTleVerifyTaskGraph
    : Pass<"triton-tle-verify-task-graph", "mlir::ModuleOp"> {
  let summary = "verify TLE mega task graph metadata";

  let description = [{
    This pass verifies the Milestone 1 task IR contract. It checks that
    ``tle.task.declare`` ops reference task grids declared in the same function,
    that task-grid names are unambiguous in graph metadata, and that affine
    read/write maps target the referenced grid rank. It performs no scheduler
    codegen and does not erase marker operations.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonTleAnalyzeTaskGraph
    : Pass<"triton-tle-analyze-task-graph", "mlir::ModuleOp"> {
  let summary = "materialize TLE mega task dependency graph metadata";

  let description = [{
    This pass consumes verified ``tle.task_grid.create`` and
    ``tle.task.declare`` metadata and materializes an explicit static
    dependency graph into a deterministic ``tle.task_graph.analysis`` function
    attribute. It computes task instances, produced grid tiles, required input
    tiles, reverse edges, initial dependency counts, and initially ready task
    instances. It performs no scheduler codegen.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonTleLowerExtractTile
    : Pass<"triton-tle-lower-extract-tile", "mlir::ModuleOp"> {
  let summary = "Lower TLE extract_tile operations";

  let description = [{
    This pass lowers tle.extract_tile operations to target-specific GPU
    operations or equivalent Triton/TritonGPU IR patterns.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonTleLowerInsertTile
    : Pass<"triton-tle-lower-insert-tile", "mlir::ModuleOp"> {
  let summary = "Lower TLE insert_tile operations";

  let description = [{
    This pass lowers tle.insert_tile operations to target-specific GPU
    operations or equivalent Triton/TritonGPU IR patterns.
  }];

  let dependentDialects = ["mlir::triton::tle::TleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}
def TleConvertArgToMemDesc
    : Pass<"tle-convert-arg-to-memdesc", "mlir::ModuleOp"> {}

def TleRemoveRedundantCopy
    : Pass<"tle-remove-redundant-copy", "mlir::ModuleOp"> {}

def TleDSLRegionInline : Pass<"tle-dslregion-inline", "mlir::ModuleOp"> {}

#endif // TRITON_TLE_PASSES
