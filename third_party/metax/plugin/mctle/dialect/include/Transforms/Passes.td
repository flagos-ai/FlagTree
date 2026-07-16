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
                           "mlir::triton::mctle::McTleDialect"];
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
                           "mlir::triton::mctle::McTleDialect"];
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
                           "mlir::triton::mctle::McTleDialect"];
}

def TritonTleLowerExtractTile
    : Pass<"triton-tle-lower-extract-tile", "mlir::ModuleOp"> {
  let summary = "Lower TLE extract_tile operations";

  let description = [{
    This pass lowers tle.extract_tile operations to target-specific GPU
    operations or equivalent Triton/TritonGPU IR patterns.
  }];

  let dependentDialects = ["mlir::triton::mctle::McTleDialect",
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

  let dependentDialects = ["mlir::triton::mctle::McTleDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif // TRITON_TLE_PASSES
