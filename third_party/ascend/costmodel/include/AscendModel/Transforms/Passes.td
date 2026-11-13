//===- Passes.td - AscendModel transformation passes -------*- tablegen -*-===//
//
// This file declares the passes for the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_TRANSFORMS_PASSES_TD
#define ASCEND_MODEL_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ConvertTritonToAscend
//===----------------------------------------------------------------------===//

def ConvertTritonToAscendPass : Pass<"convert-triton-to-ascend", "ModuleOp"> {
  let summary = "Convert Triton IR to AscendModel IR";
  let description = [{
    This pass converts Triton dialect operations to AscendModel dialect
    operations for performance modeling on Ascend 910B hardware.
  }];
  let dependentDialects = [
    "mlir::ascend::AscendModelDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// InsertDataTransfers
//===----------------------------------------------------------------------===//

def InsertDataTransfersPass : Pass<"insert-data-transfers", "ModuleOp"> {
  let summary = "Insert memory transfer operations between Cube and Vector";
  let description = [{
    This pass analyzes data dependencies between Cube (matrix) and Vector
    (SIMD) operations and inserts the necessary memory transfer operations.

    On Ascend 910B, Cube and Vector have separate memory hierarchies:
    - Cube: HBM → L1 → L0A/L0B → Cube → L0C → HBM
    - Vector: HBM → UB → Vector → UB → HBM

    When data flows between them, explicit transfers are needed:
    - Vector → Cube: insert vector_store (UB→HBM) + cube_load (HBM→L1)
    - Cube → Vector: insert cube_store (L0C→HBM) + vector_load (HBM→UB)
  }];
}

//===----------------------------------------------------------------------===//
// AssignOpIDs
//===----------------------------------------------------------------------===//

def AssignOpIDsPass : Pass<"assign-op-ids", "ModuleOp"> {
  let summary = "Assign unique IDs to operations";
  let description = [{
    This pass assigns unique operation IDs to all AscendModel operations
    for tracking and scheduling purposes.
  }];
}

//===----------------------------------------------------------------------===//
// EstimateCycles
//===----------------------------------------------------------------------===//

def EstimateCyclesPass : Pass<"estimate-cycles", "ModuleOp"> {
  let summary = "Estimate execution cycles for operations";
  let description = [{
    This pass estimates the execution cycles for each AscendModel operation
    based on hardware parameters from JSON config file.

    For loops with dynamic bounds, provide bindings via arg-bindings option:

    Example for Triton kernels (options are space-separated):
      inproc-costmodel --estimate-cycles="arg-bindings=arg2=100,pid_x=0" input.mlir

    Supported binding formats:
      - argN=value or N=value  : function argument %argN
      - pid_x=value            : tt.get_program_id x
      - pid_y=value            : tt.get_program_id y
      - pid_z=value            : tt.get_program_id z

    Alternatively, use loop-trip-counts to directly override trip counts:
      inproc-costmodel --estimate-cycles="loop-trip-counts=4,6588" input.mlir

    To use custom hardware config:
      inproc-costmodel --estimate-cycles="hardware-config=path/to/config.json" input.mlir

    Multiple options (use space, not comma, to separate):
      inproc-costmodel --estimate-cycles="hardware-config=config.json arg-bindings=pid_x=0" input.mlir
  }];

  let options = [
    Option<"argBindingsStr", "arg-bindings", "std::string",
           /*default=*/"\"\"",
           "Bindings for args and program_ids (e.g., 'arg2=100,pid_x=0')">,
    Option<"loopTripCountsStr", "loop-trip-counts", "std::string",
           /*default=*/"\"\"",
           "Direct loop trip count overrides in order (e.g., '4,6588')">,
    Option<"hardwareConfigPath", "hardware-config", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration JSON file">
  ];
}

//===----------------------------------------------------------------------===//
// PipelineAnalysis
//===----------------------------------------------------------------------===//

def PipelineAnalysisPass : Pass<"analyze-pipeline", "ModuleOp"> {
  let summary = "Analyze pipeline scheduling";
  let description = [{
    This pass performs pipeline scheduling analysis for AscendModel operations,
    considering hardware unit parallelism and dependencies.

    For loops with dynamic bounds (options are space-separated):
      inproc-costmodel --analyze-pipeline="arg-bindings=arg2=100,pid_x=0" input.mlir

    To use custom hardware config:
      inproc-costmodel --analyze-pipeline="hardware-config=path/to/config.json" input.mlir

    Multiple options (use space, not comma, to separate):
      inproc-costmodel --analyze-pipeline="hardware-config=config.json arg-bindings=pid_x=0" input.mlir
  }];

  let options = [
    Option<"argBindingsStr", "arg-bindings", "std::string",
           /*default=*/"\"\"",
           "Bindings for args and program_ids (e.g., 'arg2=100,pid_x=0')">,
    Option<"loopTripCountsStr", "loop-trip-counts", "std::string",
           /*default=*/"\"\"",
           "Direct loop trip count overrides in order (e.g., '4,6588')">,
    Option<"hardwareConfigPath", "hardware-config", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration JSON file">
  ];
}

//===----------------------------------------------------------------------===//
// HIVMAnalysis
//===----------------------------------------------------------------------===//

def HIVMAnalysisPass : Pass<"analyze-hivm", "ModuleOp"> {
  let summary = "Analyze native HIVM IR scheduling and synchronization";
  let description = [{
    This pass runs the HIVM-native scheduler directly on parsed HIVM IR.
    It expects `hivm.hir.*` ops to be available as native MLIR ops, and
    reports per-pipe utilization, sync costs, and critical-path timing.

    Example:
      inproc-costmodel --analyze-hivm input.npuir.mlir
      inproc-costmodel --analyze-hivm="hardware-config=configs/ascend_910b.json arg-bindings=arg10=128" input.npuir.mlir
  }];

  let options = [
    Option<"schedulerMode", "scheduler", "std::string",
           /*default=*/"\"static\"",
           "Scheduler backend: static or des">,
    Option<"argBindingsStr", "arg-bindings", "std::string",
           /*default=*/"\"\"",
           "Bindings for dynamic HIVM values (e.g., 'arg10=128,arg11=64')">,
    Option<"hardwareConfigPath", "hardware-config", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration JSON file">,
    Option<"reportFile", "report-file", "std::string",
           /*default=*/"\"\"",
           "Optional file to write the textual HIVM report to">,
    Option<"perfettoTraceFile", "perfetto-trace-file", "std::string",
           /*default=*/"\"\"",
           "Optional Perfetto-compatible JSON trace output file">
  ];
}

//===----------------------------------------------------------------------===//
// PerfReport
//===----------------------------------------------------------------------===//
def PerfReportPass : Pass<"perf-report", "ModuleOp"> {
  let summary = "Generate performance report";
  let description = [{
    This pass generates a comprehensive performance report including
    roofline analysis, utilization metrics, and bottleneck identification.
  }];
}


#endif // ASCEND_MODEL_TRANSFORMS_PASSES_TD
