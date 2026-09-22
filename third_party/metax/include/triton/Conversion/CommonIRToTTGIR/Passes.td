#ifndef TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES
#define TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToTTGIR : Pass<"convert-common-ir-to-ttgir", "mlir::ModuleOp"> {
  let summary = "convert GPU CommonIR buffers to TritonGPU descriptors";
  let options = [Option<"enableAsyncCopy", "enable-async-copy", "bool", "false",
                        "Use native async global-to-shared copies">];
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

def CommonIRInjectDotPlan : Pass<"commonir-inject-dot-plan", "mlir::ModuleOp"> {
  let summary = "inject a chosen dot layout plan, replacing accelerate-matmul selection";
  let description = [{
    Autolayout candidate compilation uses this pass to apply a
    {mma, operandA, operandB} plan for each tt.dot. The default compiler path
    continues to use the target accelerate-matmul pass.
    A supplied dot-id mapping is validated against the current candidate domains
    before any dot is changed. Without a mapping, each dot uses its first plan.
    The pass converts operands and accumulator to the selected encodings and
    converts the result back to its original encoding. Subsequent layout
    conversion removal folds the introduced conversions.
  }];
  let options = [
    Option<"computeCapability", "compute-capability", "int32_t", "80",
           "target compute capability driving the MMA layout choice">,
    Option<"numWarps", "num-warps", "int32_t", "4",
           "number of warps per CTA">,
  ];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif
