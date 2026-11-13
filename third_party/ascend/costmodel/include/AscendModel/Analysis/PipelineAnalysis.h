//===- PipelineAnalysis.h - Pipeline Scheduling Analysis --------*- C++ -*-===//
//
// This file declares the pipeline analysis for Ascend NPU performance modeling.
// The analysis uses HardwareConfig for configurable hardware parameters.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_ANALYSIS_PIPELINE_ANALYSIS_H
#define ASCEND_MODEL_ANALYSIS_PIPELINE_ANALYSIS_H

#include "AscendModel/HardwareConfig.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// Pipeline Stage Representation
//===----------------------------------------------------------------------===//

/// Represents a single operation in the pipeline.
struct PipelineOp {
  int64_t opId;       // Unique operation ID
  HWUnit hwUnit;      // Hardware unit (cube, vector, mte2, etc.)
  int64_t startCycle; // Scheduled start cycle
  int64_t duration;   // Duration in cycles
  int64_t endCycle;   // Computed end cycle
  Operation *mlirOp;  // Reference to MLIR operation
  std::string opName; // Human-readable name

  // Performance attributes from HardwareConfig
  int64_t bytes; // Memory transfer bytes (for MTE operations)
  int64_t flops; // FLOPs (for compute operations)

  // Loop handling
  int64_t loopMultiplier; // Number of times this op executes (due to loops)

  // Dependencies
  llvm::SmallVector<int64_t, 4> dependsOn; // Op IDs this depends on

  PipelineOp()
      : opId(-1), hwUnit(HWUnit::Scalar), startCycle(0), duration(0),
        endCycle(0), mlirOp(nullptr), bytes(0), flops(0), loopMultiplier(1) {}
};

//===----------------------------------------------------------------------===//
// Hardware Unit Pipeline
//===----------------------------------------------------------------------===//

/// Represents the pipeline for a single hardware unit.
/// In Ascend 910B, all units are fully pipelined and can run in parallel.
class HWUnitPipeline {
public:
  HWUnitPipeline(HWUnit unit) : unit(unit), currentCycle(0) {}

  /// Schedule an operation on this unit.
  /// Returns the actual start time (may be later than earliestStart if unit is
  /// busy).
  void scheduleOp(PipelineOp &op, int64_t earliestStart);

  /// Get the next available cycle for scheduling.
  int64_t getNextAvailableCycle() const { return currentCycle; }

  /// Get all scheduled operations.
  const std::vector<PipelineOp *> &getScheduledOps() const {
    return scheduledOps;
  }

  /// Get total busy cycles.
  int64_t getTotalBusyCycles() const;

  /// Get utilization percentage.
  double getUtilization(int64_t totalCycles) const;

  HWUnit getUnit() const { return unit; }

private:
  HWUnit unit;
  int64_t currentCycle;
  std::vector<PipelineOp *> scheduledOps;
};

//===----------------------------------------------------------------------===//
// Dependency Graph
//===----------------------------------------------------------------------===//

/// Manages operation dependencies for scheduling.
class DependencyGraph {
public:
  /// Add an operation to the graph.
  void addOp(int64_t opId, Operation *op);

  /// Add a dependency edge.
  void addDependency(int64_t fromId, int64_t toId);

  /// Get all operations that must complete before given op.
  llvm::SmallVector<int64_t, 4> getDependencies(int64_t opId) const;

  /// Compute topological order for scheduling.
  std::vector<int64_t> getTopologicalOrder() const;

  /// Check if graph has cycles.
  bool hasCycle() const;

private:
  llvm::DenseMap<int64_t, Operation *> ops;
  llvm::DenseMap<int64_t, llvm::SmallVector<int64_t, 4>> edges;
  llvm::DenseMap<int64_t, llvm::SmallVector<int64_t, 4>> reverseEdges;
};

//===----------------------------------------------------------------------===//
// Pipeline Scheduler
//===----------------------------------------------------------------------===//

/// Main scheduler for Ascend NPU pipeline analysis.
/// Uses HardwareConfig for configurable hardware parameters.
///
/// Key insight: In Ascend 910B, all hardware units are fully pipelined:
/// - Cube path: cube_mte2 | mte1 | cube | fixpipe can ALL run in parallel
/// - Vector path: vector_mte2 | vector | mte3 can ALL run in parallel
/// - Cube path and Vector path can run in parallel
///
/// The only constraints are DATA DEPENDENCIES between operations.
class PipelineScheduler {
public:
  /// Construct with optional hardware config (uses default 910B if null).
  PipelineScheduler(const HardwareConfig *config = nullptr);

  /// Add an operation to be scheduled.
  void addOperation(PipelineOp op);

  /// Add a dependency between operations.
  void addDependency(int64_t fromId, int64_t toId);

  /// Perform scheduling using ASAP (As Soon As Possible) algorithm.
  /// Operations are scheduled as early as possible given:
  /// 1. Data dependencies (must wait for producers)
  /// 2. Hardware unit availability (same unit can only run one op at a time)
  bool schedule();

  /// Get scheduled operations for a specific hardware unit.
  const HWUnitPipeline &getPipeline(HWUnit unit) const;

  /// Get total execution time in cycles (single-program critical path,
  /// no scalar overhead applied).
  int64_t getTotalCycles() const { return totalCycles; }

  /// Get estimated kernel wall-clock cycles accounting for:
  ///   1. Scalar overhead (loop control, pipe_barrier sync) via
  ///      HardwareConfig::getAIVScalarOverheadFactor().
  ///   2. Wave serialisation: ceil(numPrograms / numParallelUnits) waves.
  ///
  /// numPrograms  – total programs launched (e.g. Z*H*ceil(N_CTX/BLOCK_M)).
  /// numParallelUnits – hardware cores executing in parallel (e.g. N_AIV).
  /// numInnerIters – inner-loop iterations per program (for pipe_barrier cost).
  int64_t getKernelCycles(int64_t numPrograms, int64_t numParallelUnits,
                          int64_t numInnerIters = 0) const;

  /// Get all operations in scheduled order.
  const std::vector<PipelineOp> &getAllOps() const { return operations; }

  /// Get hardware config.
  const HardwareConfig &getConfig() const { return *hwConfig; }

  /// Print timeline visualization.
  void printTimeline(llvm::raw_ostream &os) const;

  /// Print utilization report.
  void printUtilizationReport(llvm::raw_ostream &os) const;

private:
  const HardwareConfig *hwConfig;
  bool ownsConfig;

  std::vector<PipelineOp> operations;
  DependencyGraph depGraph;

  // Pipelines for each hardware unit
  std::map<HWUnit, HWUnitPipeline> pipelines;

  int64_t totalCycles;

  /// Get the earliest start time considering dependencies.
  int64_t getEarliestStartTime(const PipelineOp &op);

  /// Initialize hardware pipelines based on config.
  void initPipelines();
};

//===----------------------------------------------------------------------===//
// Performance Report
//===----------------------------------------------------------------------===//

/// Holds the complete performance analysis results.
struct PerformanceReport {
  // Hardware info
  std::string hardwareName;
  double clockFreqGHz;

  // Timing information
  int64_t totalCycles; // Critical-path cycles (single program, no overhead)
  double totalTimeUs;  // Corresponding wall time
  int64_t kernelTotalCycles = 0; // With scalar overhead + wave serialisation
  int64_t numWaves = 0;          // ceil(numPrograms / numParallelUnits)

  // Per-unit utilization
  std::map<HWUnit, double> unitUtilization;
  std::map<HWUnit, int64_t> unitBusyCycles;

  // Bottleneck analysis
  HWUnit bottleneckUnit;
  double bottleneckUtilization;

  // Roofline metrics
  double arithmeticIntensity; // FLOP/Byte
  double achievedTFLOPS;
  double peakTFLOPS;
  double achievedBandwidth; // GB/s
  double peakBandwidth;     // GB/s (from HardwareConfig)
  bool isComputeBound;

  // Detailed operation breakdown
  struct OpStat {
    std::string opType;
    int64_t count;
    int64_t totalCycles;
    double percentage;
  };
  std::vector<OpStat> opStats;

  /// Print the report.
  void print(llvm::raw_ostream &os) const;

  /// Export as JSON.
  std::string toJSON() const;
};

//===----------------------------------------------------------------------===//
// Roofline Analyzer
//===----------------------------------------------------------------------===//

/// Performs roofline model analysis using HardwareConfig.
class RooflineAnalyzer {
public:
  RooflineAnalyzer(const PipelineScheduler &scheduler);

  /// Analyze and generate performance report.
  PerformanceReport analyze();

  /// Get arithmetic intensity.
  double getArithmeticIntensity() const;

  /// Check if compute-bound or memory-bound.
  bool isComputeBound() const;

  /// Get theoretical peak performance.
  double getTheoreticalPeak() const;

  /// Get achieved performance.
  double getAchievedPerformance() const;

private:
  const PipelineScheduler &scheduler;
  const HardwareConfig &config;

  int64_t totalFLOPs;
  int64_t totalBytes;

  void computeMetrics();
};

} // namespace ascend
} // namespace mlir

#endif // ASCEND_MODEL_ANALYSIS_PIPELINE_ANALYSIS_H
