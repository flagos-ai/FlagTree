#ifndef TRITONMUSAGPU_PASSES
#define TRITONMUSAGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonMUSAGPUPipeline
    : Pass<"tritonmusa-pipeline", "mlir::ModuleOp"> {
  let summary = "MUSA backend-local software pipeline";
  let description = [{
    Run the MUSA backend-local software pipeliner:
    lower and expand scheduled loops, classify SQMMA accumulator pipelines
    under MUSA-specific async/wait semantics, schedule generic async waits,
    and pipeline TMA stores without routing through the public NVIDIA-only
    post-expand hooks.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"dumpIntermediateSteps", "dump-intermediate-steps",
           "bool", /*default*/"false",
           "dump intermediate pipeline steps">
  ];
}

def TritonMUSAGPUAccelerateMatmul
    : Pass<"tritonmusa-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";
  let description = [{
    Optimize the input/output layout of `dot` instructions to make them
    compatible with MUSA WMMA/SQMMA instructions.
  }];
  let dependentDialects = [];
}

def TritonMUSAGPUOptimizeDotOperands
    : Pass<"tritonmusa-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "Normalize MUSA dot operands and descriptor-fed landing views";
  let description = [{
    Apply the subset of dot-operand shared-memory canonicalizations that are
    valid on MUSA, and normalize descriptor-fed tensor transposes/reshapes into
    memdesc view ops so descriptor landing stays canonical before descriptor
    encoding and TME/pipeline lowering.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUTMELowering
    : Pass<"tritonmusa-tme-lowering", "mlir::ModuleOp"> {
  let summary = "Lower descriptor load/store to MUSA TME async copy ops";
  let description = [{
    Rewrite `tt.descriptor_load/store` to MUSA async TME operations plus
    barrier/store synchronization ops. This pass only handles descriptor ops
    and keeps descriptor semantics explicit in TTGIR.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUIssueBarrierInsertion
    : Pass<"tritonmusa-issue-barrier-insertion", "mlir::ModuleOp"> {
  let summary = "Insert MUSA TTGIR issue barriers before selected ops";
  let description = [{
    Materialize shared-memory producer-consumer barriers in TTGIR before
    MUSA TME store issue and non-dual-TME SQMMA issue sites. The pass keeps
    barrier placement explicit so LLVM lowering can remain instruction-
    selection focused.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUFinalizeBarriers
    : Pass<"tritonmusa-finalize-barriers", "mlir::ModuleOp"> {
  let summary = "Finalize MUSA async barrier resources";
  let description = [{
    Canonicalize barrier resource reservations into a single entry block
    `ttmg.bar_record` per function, while preserving the highest reserved
    barrier id and implicit async-commit-group floor semantics.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUOptimizeSqmmaAccumulatorLayout
    : Pass<"tritonmusa-optimize-sqmma-accumulator-layout", "mlir::ModuleOp"> {
  let summary = "Sink post-TME SQMMA accumulator layout conversions";
  let description = [{
    Canonicalize loop-carried SQMMA accumulator convert_layout chains that are
    introduced by descriptor/TME lowering so accumulators stay in MMA layout
    through the loop when legal.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUCanonicalizeSqmmaResultConversions
    : Pass<"tritonmusa-canonicalize-sqmma-result-conversions", "mlir::ModuleOp"> {
  let summary = "Canonicalize SQMMA result convert/trunc chains";
  let description = [{
    Push floating-point truncation through convert_layout on SQMMA results so
    result-side layout cleanup stays outside of lowering passes.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUConvertSqmmaToMTGPU
    : Pass<"tritonmusa-convert-sqmma-to-mtgpu", "mlir::ModuleOp"> {
  let summary = "Convert loop-carried SQMMA accumulator chains to MTGPU";
  let description = [{
    Rewrite guarded SQMMA hot loops from tensor-carried ttmg.squad_dot chains
    to backend-private mtgpu.sqmma chains so compact accumulator carriers stay
    alive across scf.for boundaries without leaking into public interfaces.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::triton::musa::MUSADialect",
    "mlir::triton::mtgpu::MTGPUDialect"
  ];
}

def TritonMUSAGPUOptimizeAccumulatorInit
    : Pass<"tritonmusa-optimize-accumulator-init", "mlir::ModuleOp"> {
  let summary = "Optimize MUSA dot accumulator zero-init to first-use useC";
  let description = [{
    Convert explicit zero-initialized MUSA dot accumulators into first-use
    useC flags for loop-carried SQMMA/WMMA patterns when this is legal under
    current MUSA lowering semantics.
  }];
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUOptimizeDescriptorEncoding
    : Pass<"tritonmusa-optimize-descriptor-encoding", "mlir::ModuleOp"> {
  let summary = "Optimize descriptor encodings for MUSA TTGIR pipelines";
  let description = [{
    Canonicalize descriptor encodings before MUSA TME lowering.
    This pass is intentionally backend-local so MUSA pipelines do not depend
    on NVIDIA transform pass entry points.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::musa::MUSADialect"
  ];
}

def TritonMUSAGPUMarkInplaceLoads
    : Pass<"tritonmusa-mark-inplace-loads", "mlir::ModuleOp"> {
  let summary = "Mark TTGIR loads that feed same-address stores";
  let description = [{
    Mark Triton load operations whose pointer expression is structurally
    equivalent to a store pointer in the same function. This pass is used by
    the PH1 backend-local inplace load optimization and intentionally stays in
    the transform pipeline so LLVM lowering remains selection focused.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // TRITONMUSAGPU_PASSES
