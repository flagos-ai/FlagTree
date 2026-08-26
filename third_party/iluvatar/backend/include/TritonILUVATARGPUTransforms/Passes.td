#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonILUVATARGPUAccelerateMatmul : Pass<"tritoniluvatargpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., ILUVATAR tensor cores)
  }];

  let constructor = "mlir::createTritonILUVATARGPUAccelerateMatmulPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">,
    Option<"useSme", "use-sme",
           "uint32_t", /*default*/"0",
           "this load can use SME">
  ];
}

def TritonILUVATARGPUSmeLoad : Pass<"tritoniluvatargpu-sme-load", "mlir::ModuleOp"> {
  let summary = "sme load";

  let description = [{
    Optimize the block layout of `load` instruction to make them compatible hardware accelerators
    (e.g., MR SME)
  }];

  let constructor = "mlir::createTritonILUVATARGPUSmeLoadPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">,
  ];
}

def TritonILUVATARGPUMMAReduceThreadLocality : Pass<"tritoniluvatargpu-mma-reduce-thread-locality", "mlir::ModuleOp"> {
  let summary = "defer cross-lane warp reduce of #mma loop-carried reductions";

  let description = [{
    Iluvatar-specific sibling of `tritongpu-optimize-thread-locality` that
    handles `#mma` reductions (e.g. FlashAttention online-softmax running sum).
    It splits the register-resident part of the reduce axis into a trailing
    dimension via a free view, does the thread-local reduce in the loop, carries
    the partial (rescaled by the softmax `alpha` each iteration), and performs
    the single cross-lane reduce once after the loop.
  }];

  let constructor = "mlir::createTritonILUVATARGPUMMAReduceThreadLocalityPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonILUVATARGPUOptimizeEpilogue : Pass<"tritoniluvatargpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonILUVATARGPUOptimizeEpiloguePass()";

  let dependentDialects = [];
}

def TritonILUVATARGPUUpdateAsyncWaitCount : Pass<"tritoniluvatargpu-update-async-wait-count", "mlir::ModuleOp"> {
  let summary = "Convert async wait group counts to outstanding SME G2S intrinsic counts";

  let description = [{
    Gluon/TTGIR `ttg.async_wait {num}` counts outstanding `ttg.async_commit_group`s.
    Iluvatar lowering needs `G2S_CNT` (number of outstanding `llvm.bi.sme.load`
    intrinsics). This pass walks backwards from each wait, sums SME tile-load
    costs for the requested outstanding groups, and rewrites `num` in place
    (marking the op with `ttg.iluvatar.g2sWaitCnt`).
  }];

  let constructor = "mlir::createTritonILUVATARGPUUpdateAsyncWaitCountPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::scf::SCFDialect"];
}

#endif
