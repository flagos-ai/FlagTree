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

def TritonILUVATARGPUChainDotKRotate : Pass<"tritoniluvatargpu-chain-dot-krotate", "mlir::ModuleOp"> {
  let summary = "make a chain dot's #mma -> #dot_op<opIdx=1> convert an identity";

  let description = [{
    A chain dot consumes a previous dot's accumulator as its B operand, so the
    value arrives in the accumulator (`#mma`) tile but the TCU wants the operand
    (`#dot_op`) tile. The two tiles differ only in the placement of the low three
    bits of the K axis, and K is the axis the dot reduces over, so relabeling it
    on both operands leaves the result unchanged. This pass sets `kRotate` on both
    operands of such a dot, which selects that relabeling in `iluvatarDotTile` and
    collapses the convert to an identity -- saving the ~17 warp shuffles per tile
    that the generic conversion costs on every iteration of a flash-attention KV
    loop. It only fires when the rotated operand layout is bit-for-bit equal to
    the producing `#mma` layout.
  }];

  let constructor = "mlir::createTritonILUVATARGPUChainDotKRotatePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonILUVATARGPUFAPipeline : Pass<"tritoniluvatargpu-fa-pipeline", "mlir::ModuleOp"> {
  let summary = "prefetch the K tile of a flash-attention KV loop one iteration ahead";

  let description = [{
    SmeLoad stages both dot operands of a flash-attention KV loop through shared
    memory, but it leaves each `tt.load` right in front of the dot that consumes
    it. For the K tile that means the SME global->shared transfer is issued and
    then immediately waited on, so none of its latency is hidden.

    This pass gives K a single mutable buffer allocated before the loop and
    seeded with the first tile. Each iteration reads the tile the previous one
    fetched and, once the QK dot has consumed it, issues the fetch for the next
    iteration into the same buffer -- so the transfer overlaps the softmax and
    the PV dot instead of stalling in front of the MMA. On the last iteration
    the address is clamped to the current tile, which keeps the (dead) fetch in
    bounds.

    K-prefetch rewrite gates (same as v3.2 IluvatarFlashAttentionPipeline):
    - only f16/bf16 (fp32 would pin K across the loop and overflow 128KB shared
      on BLOCK 128x128)
    - only when num_stages <= 1 (otherwise the generic software pipeliner owns
      the loop)
    - `TRITON_ILUVATAR_MR_FA_PIPELINE_OPT` kill switch (default on)

    MR (compute capability == 71) is enforced by the backend when scheduling
    this pass, not inside the rewrite.
  }];

  let constructor = "mlir::createTritonILUVATARGPUFAPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"1",
           "global num_stages; K-prefetch runs only when this is <= 1">,
  ];
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
