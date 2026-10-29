#ifndef TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES
#define TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToTTGIR : Pass<"convert-common-ir-to-ttgir", "mlir::ModuleOp"> {
  let summary = "convert GPU CommonIR buffers to TritonGPU descriptors";
  let description = [{
    Converts the target-independent CommonIR produced by tle.gpu primitives into
    the MetaX TritonGPU descriptor form. In particular, tile.alloc becomes
    ttg.local_alloc with a ttg.memdesc result. Temporary CommonIR buffer bridges
    are resolved to that descriptor before existing TLE operations are lowered.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

def CommonIRLowerTensorTiles
    : Pass<"commonir-lower-tensor-tiles", "mlir::ModuleOp"> {
  let summary = "lower CommonIR tensor tiles after TTGIR layout assignment";
  let description = [{
    Lowers static tensor tile indices to MetaX extract_tensor and insert_tensor.
    For blocked tensors, inserts layout conversions when the original thread
    distribution cannot represent the logical tile as register selection.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

def CommonIRPlanSharedLayouts
    : Pass<"commonir-plan-shared-layouts", "mlir::ModuleOp"> {
  let summary = "plan shared buffers for already assigned MMA operands";
  let description = [{
    Uses existing MMA operands and writer continuity to choose shared layouts
    for ordinary FP16/BF16 local loads. Updates allocations and their views,
    preserving explicit shared layouts. Runs after Coalesce and before tensor
    layout propagation; it does not select MMAs or rewrite register layouts.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIRPropagateLayouts
    : Pass<"commonir-propagate-layouts", "mlir::ModuleOp"> {
  let summary = "reconcile memory layouts through logical tensor views";
  let description = [{
    Collects legal async-copy operands and the coalesced layouts of ordinary
    loads feeding shared stores. Uses CommonIR tensor propagation to reconcile
    these requirements through addresses, logical tiles, transposes and loops.
    Preserves explicit tensor constraints and shared layouts, leaving conversions
    at incompatible boundaries. Runs after Coalesce and shared planning, before
    CommonIRLowerTensorTiles; it does not select MMAs or move global loads.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

def CommonIRLegalizeAsyncCopy
    : Pass<"commonir-legalize-async-copy", "mlir::ModuleOp"> {
  let summary = "adapt async-copy operands to native MetaX shared addressing";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIRPrepareBsmLayouts
    : Pass<"commonir-prepare-bsm-layouts", "mlir::ModuleOp"> {
  let summary = "prepare MetaX split-load/BSM dot and shared layouts";
  let description = [{
    Derives a default MMA for dots fed by a split local_load and bsm_perm,
    plans their shared buffer views, and assigns the matching register layouts.
    Runs before AccelerateMatmul; ordinary dots keep the existing selector.
  }];
  let options =
      [Option<"computeCapability", "compute-capability", "int32_t", "80",
              "target compute capability">,
       Option<"numWarps", "num-warps", "int32_t", "4", "warps per CTA">];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif
