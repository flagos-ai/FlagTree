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

#endif // MUSA_TLE_TRANSFORMS_PASSES
