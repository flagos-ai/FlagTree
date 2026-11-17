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
  let summary = "select shared layouts and compatible copy operand layouts";
  let description = [{
    Optimizes unspecified shared layouts for FP16/BF16 MMA operands using
    writer continuity. Keeps allocations unchanged on conflicting or unsupported
    requirements. Adapts copy/store inputs with convert_layout, including copies
    to fixed/default buffers. Runs after Coalesce and preserves explicit layouts.
    Reports async copies whose fixed layouts or operands cannot be supported.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def CommonIROptimizeTensorTileLayouts
    : Pass<"commonir-optimize-tensor-tile-layouts", "mlir::ModuleOp"> {
  let summary = "reduce layout conversions around logical tensor tiles";
  let description = [{
    Propagates existing conversion targets through logical tensor tiles, pure
    producers and loop carriers. Preserves explicit layouts and memory operations;
    unsupported cases keep their conversions. Runs before tensor tile lowering.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

#endif
