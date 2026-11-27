#ifndef TRITONXPU_PASSES
#define TRITONXPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Functionality Pass
//===----------------------------------------------------------------------===//

def TritonXPUCreateGM2LM : Pass<"tritonxpu-create-gm2lm", "mlir::ModuleOp"> {
  let summary = "Create GM2LM for XPU.";

  let description = [{
    tt.load => triton_xpu.gm2lm + triton_xpu.load;
    tt.store => triton_xpu.store + triton_xpu.lm2gm.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"xpuArch", "xpu-arch", "uint32_t", /*default*/ "3",
                        "XPU architecture">,
                 Option<"atomicSim", "atomic-sim", "bool", /*default*/ "1",
                        "Atomic Simulation">,
                 Option<"oneCoreActOnly", "core0-act-only", "bool",
                        /*default*/ "0", "Core0 Calculation Activation Only">,
                 Option<"isUseMaskZero", "is-use-mask-zero", "bool",
                        /*default*/ "0", "Whether Use Mask Zero Method">,
  ];
}

def TritonXPUTLELegalize : Pass<"tritonxpu-tle-legalize", "mlir::ModuleOp"> {
  let summary =
      "Legalize for XPU TLE kernels (separate from tritonxpu-legalize).";

  let description = [{
    TLE-specific legalize pass. Duplicated from tritonxpu-legalize so that TLE
    kernels can be legalized independently without affecting the shared
    (non-TLE) legalize behavior.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPULegalize : Pass<"tritonxpu-legalize", "mlir::ModuleOp"> {
  let summary = "Legalize for XPU.";

  let description = [{
    Insert scf.for when local memory is smaller than block size.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"bufferSize", "lm-buflen", "uint32_t",
                        /*default*/ "128", "buffer size for local memory">,
                 Option<"coreNum", "core-num", "uint32_t", /*default*/ "64",
                        "core num">,
                 Option<"groupsPerCluster", "groups-per-cluster", "uint32_t",
                        /*default*/ "1", "groups per cluster">,
                 Option<"isUseMaskZero", "is-use-mask-zero", "bool",
                        /*default*/ "0", "Whether Use Mask Zero Method">,
  ];
}

def TritonXPUTLECoreTiling : Pass<"tritonxpu-tle-core-tiling", "mlir::ModuleOp"> {
  let summary = "Core tiling for XPU TLE kernels (2D reduce: row-tiled or large-N).";

  let description = [{
    Core-tiling pass for the XPU TLE pipeline. The default ClusterLayout splits
    the last dim across all cores, forcing a cross-core reduction for a 2D
    row-reduce (axis=1) and pinning the row block to 1. This pass retags the 2D
    cluster tensors so the work maps cleanly onto the core grid via a single
    parameterized ClusterLayout (getTiled2D) with parameters (g, ngroup) derived
    from the anchor row count M and core_num, choosing one of two modes for a
    [M,N] tile feeding a tt.reduce (axis == rank-1):

      * RowTiled  (M % core_num == 0): g=1, ngroup=core_num. Distribute whole
        rows across cores (coresPerGroup=[1,1], groupsPerCluster=[core_num,1],
        sizePerCore=[ceil(M,core_num),N]); each core owns M/core_num whole rows
        and the reduce becomes fully local. When rows-per-core > 1 the layout
        uses order=[1,0] (row-major) so memory/elementwise/reduce agree.
      * LargeN    (2 <= M and core_num % M == 0, M < core_num): g=core_num/M,
        ngroup=M. Split each row across g cooperating column-cores
        (coresPerGroup=[1,g], groupsPerCluster=[M,1], sizePerCore=[1,ceil(N,g)],
        order=[0,1]); each row becomes an independent group and the reduce is
        cross-core (smem+barrier). Requires N % g == 0 (static tile col width).

    Module attrs are set to threads-per-warp == product(coresPerGroup) == g and
    num-warps == product(groupsPerCluster) == ngroup. Uses trait-driven layout
    propagation and is a safe no-op (falls back to the default layout) unless a
    dominant 2D tile satisfies one of the modes above and the propagation is
    conflict-free.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"dumpFlag", "dump-flag",
           "bool", /*default*/"0",
           "detail dump flag">,
    Option<"bufferSize", "lm-buflen",
            "uint32_t", /*default*/"128",
            "buffer size for local memory">,
    Option<"coreNum", "core-num",
            "uint32_t", /*default*/"64",
            "core num">
  ];
}

def TritonXPUTLESharedMemAlloc
    : Pass<"tritonxpu-tle-sm-alloc", "mlir::ModuleOp"> {
  let summary = "Assign static SM byte offsets to TLE cluster-shared buffers.";

  let description = [{
    Assigns a static byte offset (`xpu.sm_offset`) to every TLE cluster-shared
    (scope == smem) `ttg.local_alloc`, stamped by the frontend with
    `xpu.mem_scope = "smem"`. Such a buffer is not a per-core LM allocation: all
    cores share one copy in the module-level `global_smem` window (addrspace 2),
    and the GM->SM copy / per-core local_ptr lowerings recompute their base as
    `global_smem + sm_offset`.

    Buffers are stacked downward from the top of the 256KB SM window, each
    64B-aligned, in deterministic walk order. The lower 128KB is reserved for the
    reduce/scan scratch (which grows up from offset 0); the pass fails if the
    smem buffers together exceed the top 128KB. Runs on the TLE path only, after
    tritonxpu-tle-legalize and before the LLVM lowering.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUTLEDtypeConvert
    : Pass<"tritonxpu-tle-dtype-convert", "mlir::ModuleOp"> {
  let summary = "Promote bf16 TLE compute to f32, keeping bf16 in LM.";

  let description = [{
    The TLE counterpart of tritonxpu-dtype-convert (XPU3: BF16 => FP32), seeded
    from `tt.store` through a `tle_local_ptr` on a bf16 LM buffer.

    XPU3 has neither a bf16 ALU nor a bf16 <-> f32 convert instruction, so bf16
    compute cannot be vectorized: the arithmetic is promoted to f32 and the
    conversion is left on the LM boundary, where the load/store lowering fuses
    it (VecBF16ToFP32* / VecFP32ToBF16*). Because tt.load/tt.store require
    "result/value type == pointee type", the boundary op's pointer tensor is
    retyped to the promoted register type and the op is marked `xpu.lm_bf16` to
    record that the memory is still bf16.

    A tree whose bf16 values cross a region-carrying op (a tt.reduce combine
    body) is left untouched, i.e. it keeps computing in bf16 unvectorized.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUTLEPipeline : Pass<"tritonxpu-tle-pipeline", "mlir::ModuleOp"> {
  let summary = "Software-pipeline TLE GM->LM/SM copies (double buffering).";

  let description = [{
    Overlap a TLE loop's DMA fill with its compute by rotating the destination
    buffer through the loop's iter_args, so iteration i computes out of the buffer
    filled by iteration i-1 while the fill for i+1 is in flight.

    The XPU cluster ISA has no counting wait -- `mfence` is a bulk drain per memory
    space -- so depth is capped at 1 per staged memory space, i.e.
    `maxStages = kStagedSpaces.size() + 1`. Depth 2 therefore alternates the LM and
    SM channels across two unrolled halves of the loop.

    Body order is `tle_wait ; issue prefetch(i+1) ; compute(i)`. The reverse order
    is a structural no-op under a bulk drain -- the fence would drain the copy just
    issued -- while still looking correct in the IR.

    The local- and shared-memory budget checks run unconditionally, even with
    `enable=false`, because an over-budget allocation is a hard error regardless of
    whether pipelining is applied.

    Must run BEFORE tritonxpu-tle-legalize, which stamps unique reduce/scan ids
    that cloning the loop body would duplicate.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"enable", "enable",
           "bool", /*default*/"0",
           "Enable the pipelining rewrite (budget checks always run)">,
    Option<"numStages", "num-stages",
           "uint32_t", /*default*/"1",
           "Default pipeline depth+1 when a loop has no tt.num_stages">,
    Option<"waitMask", "wait-mask",
           "uint32_t", /*default*/"1",
           "mfence mask for the LM channel (bit0=LM, bit1=SM, bit2=GM)">,
    Option<"lmBytesLimit", "lm-bytes-limit",
           "uint32_t", /*default*/"8192",
           "Per-core local memory budget in bytes">,
  ];

}

def TritonXPUMask : Pass<"tritonxpu-mask", "mlir::ModuleOp"> {
  let summary = "Mask for Calculation.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"oneCoreActOnly", "core0-act-only", "bool",
                        /*default*/ "0", "Core0 Calculation Activation Only">,
                 Option<"isUseMaskZero", "is-use-mask-zero", "bool",
                        /*default*/ "0", "Whether Use Mask Zero Method">,
  ];
}

def TritonXPUAlloca : Pass<"tritonxpu-alloca", "mlir::ModuleOp"> {
  let summary = "Alloca buffer for gm2lm and lm2gm.";

  let description = [{
    triton_xpu.gm2lm => triton_xpu.alloca + triton_xpu.gm2lm
    triton_xpu.lm2gm => triton_xpu.alloca + triton_xpu.lm2gm
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"bufferSize", "lm-buflen", "uint32_t",
                        /*default*/ "128", "buffer size for local memory">,
                 Option<"coreNum", "core-num", "uint32_t", /*default*/ "64",
                        "core num">];
}

def TritonXPUDtypeConvert : Pass<"tritonxpu-dtype-convert", "mlir::ModuleOp"> {
  let summary = "Dtype Convert for XPU.";

  let description = [{
    XPU2: FP16 => FP32
    XPU3: BF16 => FP32
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
  let options = [Option<"xpuArch", "xpu-arch", "uint32_t", /*default*/ "3",
                        "XPU architecture">];
}

def TritonXPULoopInvariantStaging
    : Pass<"tritonxpu-loop-invariant-staging", "mlir::ModuleOp"> {
  let summary =
      "Stage a loop-invariant small-packet gm2lm out of the grid loop.";

  let description = [{
    Hoists a loop-invariant, read-only `gm2lm` out of the grid `scf.for`.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPULoopGrid : Pass<"tritonxpu-loop-grid", "mlir::ModuleOp"> {
  let summary = "Create loop on triton programs for out-of-bounds grid_size.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUPrint : Pass<"tritonxpu-print", "mlir::ModuleOp"> {
  let summary = "Lower TT_PrintOp to TTX_XPUPrintOp.";

  let description = [{}];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUUnrollControl
    : Pass<"tritonxpu-unroll-control", "mlir::ModuleOp"> {
  let summary = "Control the unroll size.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"bufferSize", "lm-buflen",
            "uint32_t", /*default*/"128",
            "buffer size for local memory">,
    Option<"coreNum", "core-num",
            "uint32_t", /*default*/"64",
            "core num">,
    Option<"isUseMaskZero", "is-use-mask-zero",
           "bool", /*default*/"0",
           "Whether Use Mask Zero Method">,
    Option<"unrollNum", "unroll-num",
           "uint32_t", /*default*/"2",
           "unroll num">,
    Option<"vrfBudget", "vrf-budget",
           "uint32_t", /*default*/"24",
           "number of vector registers that can stay live per core; "
           "calibrated against measured device time, see compiler.py">,
    Option<"budgetTiling", "budget-tiling",
           "bool", /*default*/"1",
           "Derive the tile factor from the vector register budget instead of "
           "the unroll-num knob">,
    Option<"pinUnrollNum", "pin-unroll-num",
           "int32_t", /*default*/"0",
           "What to do with the two historical unroll pins "
           "(bool-store-vectorize=4, core-deal-multi-rows=1): 0 (the default) "
           "ignores them and lets the pressure model decide, <0 restores them "
           "as an escape hatch, >0 overrides the constant so its time curve can "
           "be swept. The constraints the pins were standing in for now live in "
           "isLegalIterNum (isMultiRowLegal / isFusedI8StoreLegal), and both "
           "pinned factors measured ~19% slower than the model's pick on device "
           "(xpu-tile-redesign/reference/findings.md 1.47), so the pins have no "
           "remaining justification -- correctness, legality, or speed">,
    Option<"openDecisionEntry", "open-decision-entry",
           "bool", /*default*/"0",
           "Off by default: let the trees that hold no vector value of their "
           "own take the criteria chain's factor instead of the legacy one. "
           "Three sites in the probe suite reach that entry (bool pointwise, "
           "welford reduce, pairmax reduce) and all three move 64 -> 1, because "
           "peakVRegs is 0 there so tier 1 cannot bind and tier 3 is monotone. "
           "The measured optimum on bool is an interior point (iterNum=16), so "
           "this is a measurement knob, not a fix "
           "(xpu-tile-redesign/reference/findings.md 1.51)">,
  ];
}

//===----------------------------------------------------------------------===//
// Optimization Pass
//===----------------------------------------------------------------------===//

def TritonXPUScalarAnalysis
    : Pass<"tritonxpu-scalar-analysis", "mlir::ModuleOp"> {
  let summary = "Scalar/Vector inference via tt.make_range + use-def chain.";

  let description = [{
    Classifies each tensor SSA value as Scalar or VectorContiguous and stamps
    offsetState on gm2lm/lm2gm when the address is contig-1.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"aggressive", "aggressive", "bool", /*default=*/"false",
                        "When true, skip blockFromLoad guard and only update "
                        "ops with offsetState=-1">];
}

def TritonXPUOffsetAnalysis
    : Pass<"tritonxpu-offset-analysis", "mlir::ModuleOp"> {
  let summary = "Analysis Ptr's Offset State.";

  let description = [{
    Given buffer_size = 8

    offsets = [0, 1, 2, 3, 4, 5, 6, 7] -> Continuous
    offsets = [0, 0, 0, 0, 0, 0, 0, 0] -> DiscreteSame
    offsets = [4, 5, 6, 7, 0, 1, 2 ,3] -> Discrete
    offsets = [0, 1, 1001, 1002, 3, 4, 2001, 2002] -> Unknown
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"dumpFlag", "dump-flag", "bool", /*default*/ "0",
                        "detail dump flag">,
                 Option<"bufferSize", "lm-buflen", "uint32_t",
                        /*default*/ "128", "buffer size for local memory">,
                 Option<"isUseMaskZero", "is-use-mask-zero", "bool",
                        /*default*/ "0", "Whether Use Mask Zero Method">,
  ];
}

def TritonXPUCoreTiling : Pass<"tritonxpu-core-tiling", "mlir::ModuleOp"> {
  let summary = "Core Tiling Optimization.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options =
      [Option<"dumpFlag", "dump-flag", "bool", /*default*/ "0",
              "detail dump flag">,
       Option<"bufferSize", "lm-buflen", "uint32_t", /*default*/ "128",
              "buffer size for local memory">,
       Option<"coreNum", "core-num", "uint32_t", /*default*/ "64", "core num">,
       Option<"groupsPerCluster", "groups-per-cluster", "uint32_t",
              /*default*/ "1", "groups per cluster">,
       Option<"tritonAutoCoreTiling", "triton-auto-core-tiling", "bool",
              /*default*/ "0",
              "auto compute groupsPerCluster from largest 2d tensor">];
}

def TritonXPUNormalize : Pass<"tritonxpu-normalize", "mlir::ModuleOp"> {
  let summary = "Pre-vectorization normalization rewrites.";

  let description = [{
    Scalar-to-scalar rewrites that put the IR into the shapes the vectorizer
    matches.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"dumpFlag", "dump-flag", "bool", /*default*/ "0",
                        "detail dump flag">,
                 Option<"compareFusion", "compare-fusion", "bool",
                        /*default*/ "false", "compare fusion">];
}

def TritonXPUVectorize : Pass<"tritonxpu-vectorize", "mlir::ModuleOp"> {
  let summary = "Vectorize Calculation.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"dumpFlag", "dump-flag",
           "bool", /*default*/"0",
           "detail dump flag">,
    Option<"compareFusion", "compare-fusion",
            "bool", /*default*/"false",
            "compare fusion">,
    // Step 3.2: take the set handed to the rewrite from the Vector-Flow
    // partition's terminal state instead of the all-or-nothing closure walk.
    // Default on since step 3.2n (the gate is in XPUOptions.per_op_decision).
    Option<"perOpDecision", "per-op-decision",
            "bool", /*default*/"true",
            "decide vectorization per op from the Vector-Flow terminal state">,
    // Escape hatch for the one TLE read whose vector form is a stack bet: a
    // scope=smem buffer is cluster-shared, so vectorizing it keeps a whole
    // per-core tile live in vector registers. Cleared by the driver on the last
    // rung of the TLE retune ladder (and by TRITONXPU_IS_CLOSE_TLE_SM_VEC), so a
    // kernel that unroll control cannot slice under the budget recompiles with
    // the scalar SM read instead of failing to compile. The LM path is
    // unaffected either way.
    Option<"tleSmemVec", "tle-smem-vec",
            "bool", /*default*/"true",
            "vectorize reads of a scope=smem TLE buffer">,
    // The retune ladder's last rung. Weights in LM or in SM both end up asking
    // unroll control for stack relief, so when its knobs are spent the only
    // thing left is to not vectorize the TLE segment at all: false makes both
    // TLE arms report no coverage and the segment stays scalar. Ships a slow
    // kernel instead of no kernel.
    Option<"tleVec", "tle-vec",
            "bool", /*default*/"true",
            "vectorize the TLE compute segment at all">
  ];
}

//[FlagTree-klx-sync] restored: dropped from the #992 baseline, still
// bound by triton_xpu.cc and driven by TRITONXPU_MEMORY_ASYNC paths.
def TritonXPUAsyncLoadSchedule
    : Pass<"tritonxpu-async-load-schedule", "mlir::ModuleOp"> {
  let summary = "Move XPU loads near first user and insert explicit mfence.";

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::LLVM::XPU::LLVMXPUDialect"];

  let options = [Option<"dumpFlag", "dump-flag", "bool", /*default*/ "0",
                        "detail dump flag">,
  ];
}

def TritonXPUMemoryAsync : Pass<"tritonxpu-memory-async", "mlir::ModuleOp"> {
  let summary = "Memory Async Optimization.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"dumpFlag", "dump-flag", "bool", /*default*/ "0",
                        "detail dump flag">,
  ];
}

def TritonXPUInterleave : Pass<"tritonxpu-interleave", "mlir::ModuleOp"> {
  let summary = "Interleave for XPU.";
  let description = [{
    Convert triton.make_range triton_xpu.interleave.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUStoreControl : Pass<"tritonxpu-store-control", "mlir::ModuleOp"> {
  let summary = "Store Control for XPU.";
  let description = [{
    Only Store isCoreId0InsideGroup=0 for ReduceOp
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUOtherSim : Pass<"tritonxpu-other-sim", "mlir::ModuleOp"> {
  let summary = "Simulate Other for XPU.";
  let description = [{
    Simulate Other in LoadOp/StoreOp.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"bufferSize", "lm-buflen", "uint32_t",
                        /*default*/ "128", "buffer size for local memory">,
                 Option<"coreNum", "core-num", "uint32_t", /*default*/ "64",
                        "core num">];
}

def TritonXPUMemoryCache : Pass<"tritonxpu-memory-cache", "mlir::ModuleOp"> {
  let summary = "Memory Cache for XPU.";
  let description = [{
    Memory Cache.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"bufferSize", "lm-buflen", "uint32_t",
                        /*default*/ "128", "buffer size for local memory">,
                 Option<"coreNum", "core-num", "uint32_t", /*default*/ "64",
                        "core num">];
}

def TritonXPUMemoryInplace
    : Pass<"tritonxpu-memory-inplace", "mlir::ModuleOp"> {
  let summary = "Memory Inplace for XPU.";
  let description = [{
    Memory Inplace.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"bufferSize", "lm-buflen", "uint32_t",
                        /*default*/ "128", "buffer size for local memory">,
                 Option<"coreNum", "core-num", "uint32_t", /*default*/ "64",
                        "core num">];
}

def TritonXPUFuncConvert : Pass<"tritonxpu-func-convert", "mlir::ModuleOp"> {
  let summary = "Func Convert for XPU.";
  let description = [{
    Func Convert.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"toMlir", "to-mlir", "bool", /*default*/ "0",
                        "from triton::FuncOp to func::FuncOp or not">];
}

def TritonXPUCFToSCF : Pass<"tritonxpu-cf-to-scf", "mlir::ModuleOp"> {
  let summary = "CF To SCF for XPU.";
  let description = [{
    CF To SCF.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPULegalizeExternEW
    : Pass<"tritonxpu-legalize-extern-ew", "mlir::ModuleOp"> {
  let summary = "Legalize unsupported extern_elementwise ops for non-SDNN XPU.";
  let description = [{
    Rewrites tt.extern_elementwise ops whose symbols are not present in
    libdevice-xpu3.bc to equivalent native MLIR ops.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect",
                           "mlir::math::MathDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonXPUTileAnalysis : Pass<"tritonxpu-tile-analysis", "mlir::ModuleOp"> {
  let summary = "Report register pressure and tile geometry per block.";

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [Option<"vrfBudget", "vrf-budget", "uint32_t", /*default*/ "24",
                        "same calibrated budget the tiling decision uses">,
  ];
}

def TritonXPUVectorizabilityAnalysis
    : Pass<"tritonxpu-vectorizability-analysis", "mlir::ModuleOp"> {
  let summary = "Report which vectorization roots form a vectorizable closure.";

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"reduceVec", "reduce-vec",
           "bool", /*default*/"1",
           "match Vectorize's ReduceVec so the same roots are enumerated">,
    // Must track Vectorize's tle-smem-vec: the analysis and the rewrite have to
    // agree on which TLE reads are vectorizable, or the rewrite is handed a set
    // the analysis never validated.
    Option<"tleSmemVec", "tle-smem-vec",
           "bool", /*default*/"true",
           "match Vectorize's tleSmemVec so the same reads are admissible">,
    Option<"tleVec", "tle-vec",
           "bool", /*default*/"true",
           "match Vectorize's tleVec so the same segments are admissible">,
    Option<"preTiling", "pre-tiling",
           "bool", /*default*/"0",
           "this instance sits ahead of CoreTiling: report the state-only "
           "answer (all-Unknown fit oracle) alongside the real one">,
  ];
}

def TritonXPUTileDecide : Pass<"tritonxpu-tile-decide", "mlir::ModuleOp"> {
  let summary = "Decide the terminal representation per root, ahead of CoreTiling.";

  let description = [{
    The decision half of M2/M3, split out of `tritonxpu-unroll-control` so it can
    be taken where the information it needs is E-independent: it runs immediately
    after the `pre-tiling` vectorizability analysis and before CoreTiling.

    Per root -- the same roots Vectorize enumerates -- it reads the Vector-Flow
    partition (built with the all-Unknown fit oracle, since sizePerCore does not
    exist yet) and records the terminal state of the root's value plus the
    pack/unpack sites that state implies. That is tier 3's input, the only tier
    answerable here: tier 1 (VRF budget) and tier 2 (LM footprint) both need peak
    pressure over a fixed per-core geometry, so they stay in
    `tritonxpu-unroll-control` and are recorded as deferred rather than guessed.

    It writes only the `triton_xpu.tile_decision` carrier and never touches the
    IR, which is what lets it be inserted or removed to diff against today's
    decision. `tritonxpu-tile-analysis` resolves and erases the carrier before
    emission, so the hit rates say whether the keys survive CoreTiling, Legalize,
    Mask, Interleave, Normalize and Vectorize.

    Off unless TRITONXPU_TILE_DECIDE=1.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"reduceVec", "reduce-vec",
           "bool", /*default*/"1",
           "match Vectorize's ReduceVec so the same roots are enumerated">,
  ];
}

#endif // TRITONXPU_PASSES
