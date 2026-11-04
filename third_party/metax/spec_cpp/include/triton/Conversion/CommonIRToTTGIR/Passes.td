#ifndef TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES
#define TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToTTGIR : Pass<"convert-common-ir-to-ttgir", "mlir::ModuleOp"> {
  let summary = "convert GPU CommonIR buffers to TritonGPU descriptors";
  let description = [{
    Converts CommonIR buffers to TritonGPU descriptors: tile.alloc becomes
    ttg.local_alloc, and buffer bridges resolve to the resulting memdesc.
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
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

def CommonIROptimizeSharedLayouts
    : Pass<"commonir-optimize-shared-layouts", "mlir::ModuleOp"> {
  let summary = "optimize shared layouts without changing the existing pipeline";
  let description = [{
    Chooses layouts for shared allocations and views feeding ordinary FP16/BF16
    local loads, using MMA operands and writer continuity. Preserves explicit
    shared layouts. Runs after Coalesce and before tensor layout propagation.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIRPropagateLayouts
    : Pass<"commonir-propagate-layouts", "mlir::ModuleOp"> {
  let summary = "reconcile memory layouts through logical tensor views";
  let description = [{
    Propagates async-copy requirements and coalesced load layouts through tensor
    views and loops. Preserves explicit constraints and shared layouts, inserting
    conversions at incompatible boundaries. Runs after Coalesce and shared
    planning, before CommonIRLowerTensorTiles.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

def CommonIRClearLayoutHints
    : Pass<"commonir-clear-layout-hints", "mlir::ModuleOp"> {
  let summary = "remove temporary layout hints after TTGIR optimization";
}

def CommonIRLegalizeAsyncCopy
    : Pass<"commonir-legalize-async-copy", "mlir::ModuleOp"> {
  let summary = "adapt async-copy operands to native MetaX shared addressing";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIRResolveBsmLayouts
    : Pass<"commonir-resolve-bsm-layouts", "mlir::ModuleOp"> {
  let summary = "resolve MetaX split-load/BSM dot and shared layouts";
  let description = [{
    Derives a default MMA for dots fed by a split local_load and bsm_perm,
    plans their shared buffer views, and assigns the matching register layouts.
    Runs before AccelerateMatmul; ordinary dots keep the existing selector.
  }];
  let options =
      [Option<"computeCapability", "compute-capability", "int32_t", "80",
              "target compute capability">,
       Option<"numWarps", "num-warps", "int32_t", "4", "warps per CTA">];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif
