// flagtree tle
#ifndef TRITON_TLE_PASSES
#define TRITON_TLE_PASSES

include "mlir/Pass/PassBase.td"

def TritonTleEarlyAssignMemorySpace : Pass<"triton-tle-early-assign-memory-space", "mlir::ModuleOp"> {
  let summary = "early assign memory space";

  let description = [{
    This pass assigns memory spaces to tensor pointers according to `tt.memory_space` attribute on tensors.
    This is done early to enable subsequent passes to make better optimization decisions based on memory spaces.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::tle::TleDialect"];
}

def TritonTleAssignLocalPointersEncoding
    : Pass<"triton-tle-assign-local-pointers-encoding", "mlir::ModuleOp"> {
  let summary = "assign shared encodings to local_pointers results";

  let description = [{
    This pass ensures `tle.local_pointers` ops produced by the TLE frontend
    carry shared-memory pointer element types and a distributed encoding so
    downstream conversions do not infer layouts during the LLVM lowering.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleInsertLocalPointerBarriers
    : Pass<"triton-tle-insert-local-pointer-barriers", "mlir::ModuleOp"> {
  let summary = "insert CTA barriers between tle.local_pointers stores and loads";

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

def TritonTleLowerAsyncLoad : Pass<"triton-tle-lower-async-load", "mlir::ModuleOp"> {
  let summary = "Lower TLE async load operations";

  let description = [{
    This pass lowers tl::LoadOp with tt.load.async to target-specific GPU operations by converting
    high-level asynchronous load operations into appropriate TritonGPU primitives.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonTleLowerTmaCopy : Pass</*cli-arg*/"triton-tle-lower-tma-copy", /*Op*/"mlir::ModuleOp"> {

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

def TleConvertArgToMemDesc : Pass<"tle-convert-arg-to-memdesc", "mlir::ModuleOp"> {}

def TleDSLRegionInline : Pass<"tle-dslregion-inline", "mlir::ModuleOp"> {}

#endif // TRITON_TLE_PASSES
