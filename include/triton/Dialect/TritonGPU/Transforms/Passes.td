#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Applies software pipelining to loops in the module based on number of stages.
    This may convert some load into asynchronous loads, and multi-buffer the data.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">
  ];
}

def TritonGPUF32DotTC : Pass<"tritongpu-F32DotTC", "mlir::ModuleOp"> {
  let summary = "3xTF32 trick";

  let description = [{
    Decompose fp32 `DotOp` instructions into 4 pointwise ops and 3 fp16 `DotOp`s
    to allow using TensorCores. See https://github.com/NVIDIA/cutlass/discussions/385
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def TritonGPUPrefetch : Pass<"tritongpu-prefetch", "mlir::ModuleOp"> {
  let summary = "prefetch";

  let description = [{
    Decompose `DotOp` instructions in loops into several finer-grained `DotOp`
    that may have their operands constructed at the end of the previous iteration
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonGPUAccelerateMatmul : Pass<"tritongpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., Nvidia tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUOptimizeDotOperands : Pass<"tritongpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "fuse transpositions";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of
    hardware-accelerated transpositions.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"hoistLayoutConversion", "hoist-layout-conversion",
           "bool", /*default*/"true",
           "whether to move conver to dot operand earlier pass elementwise ops">
  ];
}

def TritonGPUCoalesce: Pass<"tritongpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    TODO
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}


def TritonGPURemoveLayoutConversions : Pass<"tritongpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

}

def TritonGPUOptimizeThreadLocality : Pass<"tritongpu-optimize-thread-locality", "mlir::ModuleOp"> {
  let summary = "Reduce the cost of synchronization between threads in an SM";

  let description = [{
    Today, this optimizes reduction yielded by loop to be thread-local until after the loop completes.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReorderInstructions: Pass<"tritongpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReduceDataDuplication: Pass<"tritongpu-reduce-data-duplication", "mlir::ModuleOp"> {
  let summary = "Reduce data duplication in register by decomposing convert[distributed -> dotOperand] "
                "into convert[distributed -> shared -> dotOperand]";

  let description = "Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUCombineTensorSelectAndIf: Pass<"tritongpu-combine-tensor-select-and-if", "mlir::ModuleOp"> {
  let summary = "Combine tensor select and if";

  let description = "For select instruction that uses the same condidtion as the if instruction in the same block "
                    "this pass combines the select into the if instruction, making the select operands returned by the "
                    "then/else yields.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUAdvancedPipeliner : Pass<"tritongpu-advanced-pipeliner", "mlir::ModuleOp"> {
  let summary = "Advanced multi-level, multi-stage memory-compute pipelining";

  let description = [{
    Applies advanced multi-level pipelining optimization with automatic buffer analysis,
    opportunity detection, circular buffer transformation, and synchronization insertion.
    Supports multi-level pipelines (Global→Shared→Register) with configurable stages,
    async copy operations, and swizzled buffers for bank conflict reduction.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect"];

  let options = [
    Option<"globalToSharedStages", "global-to-shared-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages for Global→Shared transfers">,
    Option<"sharedToRegisterStages", "shared-to-register-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages for Shared→Register transfers">,
    Option<"enableAsyncCopy", "enable-async-copy",
           "bool", /*default*/"true",
           "enable hardware async copy operations (cp.async, TMA)">,
    Option<"enableSwizzle", "enable-swizzle",
           "bool", /*default*/"false",
           "enable swizzled addressing to reduce bank conflicts">,
    Option<"minSpeedup", "min-speedup",
           "double", /*default*/"1.05",
           "minimum expected speedup threshold (default 5%)">,
    Option<"enableWarpSpecialization", "enable-warp-specialization",
           "bool", /*default*/"false",
           "enable warp specialization for producer/consumer separation">,
    Option<"enableMultiBufferFusion", "enable-multi-buffer-fusion",
           "bool", /*default*/"false",
           "enable multi-buffer fusion for shared synchronization">
  ];
}

#endif
