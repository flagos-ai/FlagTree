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

  let options =
      [Option<"bufferSize", "lm-buflen", "uint32_t", /*default*/ "128",
              "buffer size for local memory">,
       Option<"coreNum", "core-num", "uint32_t", /*default*/ "64", "core num">,
       Option<"isUseMaskZero", "is-use-mask-zero", "bool", /*default*/ "0",
              "Whether Use Mask Zero Method">,
       Option<"unrollNum", "unroll-num", "uint32_t", /*default*/ "2",
              "unroll num">,
       Option<"vrfBudget", "vrf-budget", "uint32_t", /*default*/ "24",
              "vector-register budget used when budget-tiling is on">,
       Option<"budgetTiling", "budget-tiling", "bool", /*default*/ "0",
              "Derive the tile factor from the vector register budget">,
       Option<"pinUnrollNum", "pin-unroll-num", "int32_t", /*default*/ "-1",
              "<0 keep pinned constants, 0 drop pins, >0 override them">,
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

  let options = [Option<"dumpFlag", "dump-flag", "bool", /*default*/ "0",
                        "detail dump flag">,
                 Option<"compareFusion", "compare-fusion", "bool",
                        /*default*/ "false", "compare fusion">];
}

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

  let options =
      [Option<"reduceVec", "reduce-vec", "bool", /*default*/ "1",
              "match Vectorize's ReduceVec so the same roots are enumerated">,
       Option<"preTiling", "pre-tiling", "bool", /*default*/ "0",
              "this instance sits ahead of CoreTiling">,
  ];
}

#endif // TRITONXPU_PASSES
