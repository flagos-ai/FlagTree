// flagtree tle
#ifndef TRITON_TLE_PASSES
#define TRITON_TLE_PASSES

include "mlir/Pass/PassBase.td"

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




#endif // TRITON_TLE_PASSES
