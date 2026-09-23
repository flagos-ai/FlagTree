#ifndef TRITON_NVIDIA_COMMON_IR_TO_TTGIR_PASSES
#define TRITON_NVIDIA_COMMON_IR_TO_TTGIR_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToTTGIR : Pass<"convert-common-ir-to-ttgir", "mlir::ModuleOp"> {
  let summary = "convert GPU CommonIR buffers to TritonGPU descriptors";
  let options = [Option<"enableAsyncCopy", "enable-async-copy", "bool", "false",
                        "Use native async global-to-shared copies on SM80+">];
  let description = [{
    Converts the target-independent CommonIR produced by tle.gpu primitives into
    the NVIDIA TritonGPU descriptor form. In particular, tile.alloc becomes
    ttg.local_alloc with a ttg.memdesc result. Temporary CommonIR buffer bridges
    are resolved to that descriptor before existing TLE operations are lowered.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
}

#endif
