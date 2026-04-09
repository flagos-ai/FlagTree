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
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
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

def TleDSLRegionInline : Pass<"tle-dslregion-inline", "mlir::ModuleOp"> {}

#endif // TRITON_TLE_PASSES
