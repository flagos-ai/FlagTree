#ifndef ILUVATAR_TLE_PASSES
#define ILUVATAR_TLE_PASSES

include "mlir/Pass/PassBase.td"

def TritonIluvatarTleInsertLocalPointerBarriers
    : Pass<"triton-iluvatar-tle-insert-local-pointer-barriers", "mlir::ModuleOp"> {
  let summary = "insert CTA barriers between iluvatar_tle.local_pointers stores and loads";
  let dependentDialects = ["mlir::gpu::GPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleOptimizeLocalPointerLoads
    : Pass<"triton-iluvatar-tle-optimize-local-pointer-loads", "mlir::ModuleOp"> {
  let summary = "rewrite full-view iluvatar_tle.local_pointers loads into ttg.local_load";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

def TritonIluvatarTleOptimizeLocalPointerStores
    : Pass<"triton-iluvatar-tle-optimize-local-pointer-stores", "mlir::ModuleOp"> {
  let summary = "rewrite iluvatar_tle.local_pointers stores into ttg.local_store";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

#endif // ILUVATAR_TLE_PASSES
