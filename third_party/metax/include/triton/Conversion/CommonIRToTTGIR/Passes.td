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

def CommonIRApplyLayoutPlan
    : Pass<"commonir-apply-layout-plan", "mlir::ModuleOp"> {
  let summary = "apply joint dot, memory and shared-load layout requirements";
  let description = [{
    Collects and validates a selected candidate without modifying the IR, then
    uses a shared CommonIR plan to rewrite producers, loop carriers and layout
    conversions. Logical tensor tiles remain available until the subsequent
    CommonIRLowerTensorTiles pass. With no supplied candidate, consumes only
    existing MMA layouts and memory requirements; it does not select new MMAs.
  }];
  let options =
      [Option<"computeCapability", "compute-capability", "int32_t", "80",
              "target compute capability for candidate validation">,
       Option<"numWarps", "num-warps", "int32_t", "4", "warps per CTA">,
  ];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif
