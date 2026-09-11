#ifndef TRITON_METAX_GLUON_PASSES
#define TRITON_METAX_GLUON_PASSES

include "mlir/Pass/PassBase.td"

def TritonMETAXGPUGluonInsertRequireLayout
    : Pass<"tritonmetaxgpu-gluon-insert-require-layout", "mlir::ModuleOp"> {
  let summary = "Insert explicit MetaX/C500 Gluon layout requirements";

  let description = [{
    Debug/test adapter that discovers the deterministic fallback and then calls
    the same selected-candidate requirement inserter used by CandidateBundle.
    Production candidate construction calls insertLayoutRequirements directly
    with an already selected whole-module LayoutCandidate; it does not schedule
    this standalone pass. The adapter exists only so lit tests can inspect the
    intermediate requirement IR before propagation.
  }];

  let constructor = "mlir::createTritonMETAXGPUGluonInsertRequireLayoutPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "MetaX compute capability">
  ];
}

def TritonMETAXGPUGluonPropagateLayout
    : Pass<"tritonmetaxgpu-gluon-propagate-layout", "mlir::ModuleOp"> {
  let summary = "Propagate explicit MetaX/C500 Gluon layout requirements";

  let description = [{
    Runs the shared three-state layout analysis for the explicit
    gluon.set_auto_layout, gluon.require_layout and gluon.release_layout
    contracts already present in the selected candidate clone. Concrete
    consensus retags only Auto producers and transparent carriers. Residual
    tensor contracts become ttg.convert_layout; residual non-identity memdesc
    contracts reject the current candidate. This pass does not generate
    candidates or choose C500 MMA/shared profiles.
  }];

  let constructor = "mlir::createTritonMETAXGPUGluonPropagateLayoutPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonLegalizeRegisterSlices
    : Pass<"tritonmetaxgpu-gluon-legalize-register-slices",
           "mlir::ModuleOp"> {
  let summary = "Lower concrete Gluon register slices to verified TTG views";

  let description = [{
    Lowers gluon.extract_slice and gluon.insert_slice after their layouts are
    concrete. A LinearLayout proof selects either a zero-copy physical view or
    an explicit full-tile conversion boundary. This legalization is mandatory
    for both inferred and source-authored layouts; manual layout selection must
    not bypass dialect lowering or the physical slice contract.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonLegalizeRegisterSlicesPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonResolvePlaceholderLayouts
    : Pass<"tritonmetaxgpu-gluon-resolve-placeholder-layouts",
           "mlir::ModuleOp"> {
  let summary = "Resolve deferred Gluon placeholder layouts";

  let description = [{
    Runs after requirement propagation. It rejects residual calls, unwraps
    gluon.no_verify_encoding, resolves only remaining generic distributed Auto
    placeholders to deterministic blocked layouts using existing dialect
    inference, synchronizes function ABI, and verifies that no Auto or
    NoVerify encoding remains. It never creates MMA layouts, chooses shared
    layouts, searches candidates, or repairs a propagation conflict.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonResolvePlaceholderLayoutsPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonLegalizeC500AsyncCopyLayout
    : Pass<"tritonmetaxgpu-gluon-legalize-c500-async-copy-layout",
           "mlir::ModuleOp"> {
  let summary = "Legalize C500 Gluon async-copy source layouts";

  let description = [{
    Runs after Gluon layout propagation, when source blocked layouts and
    destination swizzled shared layouts are concrete. The pass preserves the
    downstream dot-derived shared layout and clips the async-copy source vector
    width when required by C500 contiguous shared writes.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonInsertGvmArriveBarrierShared
    : Pass<"tritonmetaxgpu-insert-gvm-arrive-barrier-shared",
           "mlir::ModuleOp"> {
  let summary = "Insert MetaX GVM arrive and shared barrier pairs for Gluon";

  let description = [{
    Infer FIFO GVM wait points from C500 async_copy_global_to_local producers
    and shared-memory consumers, then insert the minimum required gvm_arrive
    and barrier_shared operations. The analysis also protects synchronous
    local_store/local_alloc writes and buffer reuse against active readers.
    Synchronization ownership is selected per function. If the input function
    contains a user-authored gvm_arrive, its complete GVM/shared-barrier
    schedule is kept explicit and this pass performs no automatic repair in
    that function. Functions without a manual wait use conservative automatic
    inference.

    This is part of the Gluon legalization pipeline because it reasons about
    Gluon-generated async-copy/local-load schedules after C500 layout
    propagation has materialized concrete TTG layouts.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass()";

  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonVerifySynchronization
    : Pass<"tritonmetaxgpu-gluon-verify-synchronization",
           "mlir::ModuleOp"> {
  let summary = "Verify concrete MetaX Gluon GVM/shared synchronization";

  let description = [{
    Reconstructs the C500 FIFO completion, shared visibility, physical reuse,
    initialization, and CTA-uniformity state for automatically synchronized
    functions. Functions containing a user-authored gvm_arrive use explicit
    synchronization ownership and are outside this automatic verifier. The
    pass performs no repair and rejects automatic schedules that would require
    another inferred wait or barrier.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonVerifySynchronizationPass()";

  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonVerifyLayoutContracts
    : Pass<"tritonmetaxgpu-gluon-verify-layout-contracts",
           "mlir::ModuleOp"> {
  let summary = "Verify MetaX Gluon hardware layout contracts";

  let description = [{
    Verifies the concrete hardware contracts that must hold before lowering to
    LLVM: dot-fed shared staging, local_load/local_store LinearLayout
    lowerability, C500 async-copy source/destination contiguity, physical
    ttg.extract_tensor subviews, and same-encoding elementwise boundaries.
    This pass is verifier-only and does not retag, synthesize, or rewrite
    layouts.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonVerifyLayoutContractsPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gluon::GluonDialect",
  ];
}

def TritonMETAXGPUGluonLegalizeLocalDotStaging
    : Pass<"tritonmetaxgpu-gluon-legalize-local-dot-staging",
           "mlir::ModuleOp"> {
  let summary = "Legalize Gluon shared-memory tensor/MMA boundaries";

  let description = [{
    Materializes the widest complete, conflict-free blocked layout before a
    local_store whose current LinearLayout aliases a shared offset across lanes
    or warps. This pass is intentionally scheduled after generic layout
    conversion removal so that the hardware legality conversion cannot be
    optimized away as a semantically redundant tensor relayout.

    It also preserves an explicit tensor boundary for a dot operand loaded
    through a transposed memdesc. C500 cannot directly lower every composition
    of a transposed shared layout and an MMA warp-broadcast layout. The pass
    emits local_load(blocked) and convert_layout(dot).

    Synchronization is intentionally outside this pass. The final C500
    synchronization planner owns all inferred GVM/shared barriers after shared
    allocation and scheduling are stable.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonLegalizeLocalDotStagingPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def TritonMETAXGPUGluonSelectC500LayoutTransfers
    : Pass<"tritonmetaxgpu-gluon-select-c500-layout-transfers",
           "mlir::ModuleOp"> {
  let summary = "Select C500 lowerings for concrete Gluon layout transfers";

  let description = [{
    Analyzes concrete ttg.convert_layout operations after layout propagation
    and cleanup, asks the target for one complete transfer implementation,
    then materializes lowering-only attributes. The target query is local to
    the source/result physical mappings and semantic use boundary. It does not
    inspect module allocation, shared-memory capacity, kernel names, or fixed
    workload shapes, and it never changes a tensor encoding.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonSelectC500LayoutTransfersPass()";

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def TritonMETAXGPUGluonReorderInstructions
    : Pass<"tritonmetaxgpu-gluon-reorder-instructions",
           "mlir::ModuleOp"> {
  let summary = "Reorder Gluon instructions with alias-aware memory safety";

  let description = [{
    Shortens register and shared-memory live ranges using the standard Triton
    scheduling preferences, but requires an explicit dependence proof before
    moving a local_load across a region or another shared-memory operation.
    Generic MLIR AliasAnalysis is refined by Gluon shared access paths only
    when distinct logical allocations or disjoint physical intervals are
    proven. Unknown effects, synchronization operations, and unresolved alias
    relationships conservatively block motion.
  }];

  let constructor =
      "mlir::createTritonMETAXGPUGluonReorderInstructionsPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

#endif
