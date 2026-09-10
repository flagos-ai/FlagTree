#ifndef MUSA_TLE_TRANSFORMS_PASSES
#define MUSA_TLE_TRANSFORMS_PASSES

def TritonMUSAGPUTLESelectEncodings
    : Pass<"tritonmusa-tle-select-encodings", "mlir::ModuleOp"> {
  let summary = "Select MUSA encodings for local pointer users";
  let description = [{
    Select stable tensor encodings for `musa_tle.local_pointers` and
    dependent load/store users before MUSA LLVM lowering.
  }];
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLEInsertLocalPointerBarriers
    : Pass<"tritonmusa-tle-insert-local-pointer-barriers", "mlir::ModuleOp"> {
  let summary = "Insert barriers between local pointer stores and loads";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLEOptimizeLocalPointerLoads
    : Pass<"tritonmusa-tle-optimize-local-pointer-loads", "mlir::ModuleOp"> {
  let summary = "Rewrite full-view local pointer loads to ttg.local_load";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLEOptimizeLocalPointerStores
    : Pass<"tritonmusa-tle-optimize-local-pointer-stores", "mlir::ModuleOp"> {
  let summary = "Rewrite full-view local pointer stores to ttg.local_store";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLEOptimizeLocalPointerAsyncStores
    : Pass<"tritonmusa-tle-optimize-local-pointer-async-stores", "mlir::ModuleOp"> {
  let summary =
      "Rewrite gmem load plus musa_tle.local_pointers subview store into async_copy";

  let description = [{
    Match conservative staging patterns of the form:

      %v = tt.load %gptr ...
      tt.store %local_ptr_subview, %v ...

    where `%local_ptr_subview` is produced by `musa_tle.local_pointers` and
    denotes a static full view or static subview of a backing shared memdesc.
    Rewrite the pair into `ttg.async_copy_global_to_local` plus commit/wait.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLELowerExclusiveCumsum
    : Pass<"tritonmusa-tle-lower-exclusive-cumsum", "mlir::ModuleOp"> {
  let summary = "Lower musa_tle.exclusive_cumsum to Triton scan and reduce ops";

  let description = [{
    Lower ``musa_tle.exclusive_cumsum`` into ``tt.scan`` for prefix
    accumulation, subtract the original input to form the exclusive sum, and use
    an index-select ``tt.reduce`` over the inclusive result to produce the
    total sum.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLELowerSqmma
    : Pass<"tritonmusa-tle-lower-sqmma", "mlir::ModuleOp"> {
  let summary = "Lower backend-local MUSA TLE SQMMA operations";
  let description = [{
    Infer the PH1 SQMMA accumulator and shared-memory operand layouts, preserve
    explicit wait boundaries, and lower musa_tle.sqmma/sqmma_wait to the MUSA
    squad-dot dialect.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLELowerPipe
    : Pass<"tritonmusa-tle-lower-pipe", "mlir::ModuleOp"> {
  let summary = "Lower MUSA cyclic, close, and one-shot payload pipes";
  let description = [{
    Lower CTA-scoped cyclic and one-shot SPSC pipes, including static
    warp-specialized SPSC endpoints placed in the default or any worker
    partition, and cyclic named-reader SPMC pipes whose endpoints may be
    placed in the default or distinct worker partitions, with whole-field or
    field-subset reader subscriptions. Lower named-reader cyclic close
    broadcast and one-shot publication in warp-specialized execution as well
    as non-warp-specialized execution.
    Payloads may be produced by whole-field descriptor-to-shared TME
    transactions, multiple non-overlapping contiguous TME fragments of one
    field, whole-field local stores, or a combination of these transports.
    Analyze structured scf.if/scf.for lifecycle regions with uniform branch
    predicates, loop-carried stage/phase equivalence, and path-complete
    writer/reader drain windows before materializing resources. Preserve
    mutually-exclusive generation alternatives without multiplying accounting.
    Materialize pipe-owned full/empty hardware-barrier rings before lifecycle
    rewriting, or reuse a user-provided stage-indexed PENDING full ring for
    TME-bearing pipes. Preserve aggregate TME completion and local-store
    publication, validate complete field-region coverage and external-barrier
    ownership, and reject unsupported forms before rewriting.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa_tle::MUSATLEDialect",
    "mlir::triton::tle::TleDialect"
  ];
}

def TritonMUSAGPUTLEPrepareWarpSpecialize
    : Pass<"tritonmusa-tle-prepare-warp-specialize", "mlir::ModuleOp"> {
  let summary = "Validate MUSA TLE static warp specialization";
  let description = [{
    Validate the restricted default-plus-static-workers contract, assign
    declaration-ordered worker warp ranges, and publish the total launch warp
    count while retaining ttg.warp_specialize for shared-memory allocation and
    late LLVM lowering.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonMUSAGPUTLELowerWarpSpecialize
    : Pass<"tritonmusa-tle-lower-warp-specialize", "mlir::ModuleOp"> {
  let summary = "Late-lower MUSA TLE static warp specialization";
  let description = [{
    Lower warp-group barriers directly from the retained default and partition
    regions, then replace the restricted warp-specialize operation with direct
    static CFG dispatch for the default and all worker partitions.  The
    lowering uses hardware barrier ids and direct SSA captures, so it allocates
    no control shared memory.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUTLELowerBarrierAllocations
    : Pass<"tritonmusa-tle-lower-barrier-allocations", "mlir::ModuleOp"> {
  let summary = "Assign MUSA hardware ids to TLE barrier declarations";
  let description = [{
    Reserve one contiguous MUSA async-barrier range per function, materialize
    hardware initialization records for every declared TLE barrier, and replace
    logical barrier allocations and indices with i32 hardware ids.  Insert one
    CTA rendezvous after all initialization records and before every barrier
    use; for explicit warp specialization this also precedes the first
    top-level partition split.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLELowerTMETransactions
    : Pass<"tritonmusa-tle-lower-tme-transactions", "mlir::ModuleOp"> {
  let summary = "Materialize explicit MUSA TME completion transactions";
  let description = [{
    Replace temporary TLE completion metadata with one aggregate barrier
    add-transaction before every completion group and one issuer arrival after
    it.  All TME copies in the group share the same barrier and raw CTA issuer
    thread.  Standalone copies form single-member groups, preserving their
    existing behavior.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUTLELowerBarrierOperations
    : Pass<"tritonmusa-tle-lower-barrier-operations", "mlir::ModuleOp"> {
  let summary = "Lower TLE waits and arrivals to MUSA hardware barriers";
  let description = [{
    Replace backend-local TLE barrier wait and arrive operations with the
    corresponding MUSA hardware barrier operations after logical barrier
    ranges have been assigned physical ids.  Consumer arrivals remain
    convergent warp-level operations and are deliberately kept separate from
    the single-issuer TME transaction arrival path.
  }];
  let dependentDialects = [
    "mlir::triton::musa::MUSADialect",
    "mlir::triton::musa_tle::MUSATLEDialect"
  ];
}

def TritonMUSAGPUTLEFinalizeExplicitLayouts
    : Pass<"tritonmusa-tle-finalize-explicit-layouts", "mlir::ModuleOp"> {
  let summary = "Finalize MUSA TLE explicit layout constraints";
  let description = [{
    Verify that temporary explicit result and memory encodings agree with the
    final TTGPU tensor types, then remove the temporary attributes before LLVM
    lowering.  Tensor encodings and non-identity layout conversions remain the
    durable representation of the selected layouts.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // MUSA_TLE_TRANSFORMS_PASSES
