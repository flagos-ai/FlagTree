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
  let summary = "lower static MCTLE tensor tiles after TTGIR layout assignment";
  let description = [{
    Lowers static tensor tile indices to MetaX extract_tensor and insert_tensor.
    For blocked tensors, inserts layout conversions when the original thread
    distribution cannot represent the logical tile as register selection.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIROptimizeSharedLayouts
    : Pass<"commonir-optimize-shared-layouts", "mlir::ModuleOp"> {
  let summary = "optimize unspecified shared layouts for existing MMA operands";
  let description = [{
    Choose a compatible shared layout for FP16/BF16 MMA consumers when the
    allocation's views and writers support it, taking writer order into account.
    Preserve fixed layouts.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIRCoalesceAsyncCopy
    : Pass<"commonir-coalesce-async-copy", "mlir::ModuleOp"> {
  let summary = "adapt async copy inputs to the destination shared layout";
  let description = [{
    Select a compatible blocked layout for the pointer and mask operands of
    MetaX async copies. Diagnose copies unsupported by native lowering.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIROptimizeTensorTileLayouts
    : Pass<"commonir-optimize-tensor-tile-layouts", "mlir::ModuleOp"> {
  let summary = "reduce layout conversions around logical tensor tiles";
  let description = [{
    Reuse a global load's thread layout for shared stores of compatible slices.
    Propagates existing conversion targets through logical tensor tiles, pure
    producers and loop carriers. Preserves explicit layouts and memory operations;
    unsupported cases keep their conversions. Runs before tensor tile lowering.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif
