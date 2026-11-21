//===- PipelineAnalysis.cpp - Pipeline scheduling analysis ----------------===//
//
// This file implements the pipeline analysis for Ascend NPU performance
// modeling, using HardwareConfig for configurable hardware parameters.
//
// Key insight: Ascend 910B has a fully pipelined architecture where all
// hardware units can run in parallel. The only scheduling constraints are
// data dependencies between operations.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <queue>

using namespace mlir;
using namespace mlir::ascend;

//===----------------------------------------------------------------------===//
// HWUnitPipeline Implementation
//===----------------------------------------------------------------------===//

void HWUnitPipeline::scheduleOp(PipelineOp &op, int64_t earliestStart) {
  // In a fully pipelined architecture, the unit can start a new operation
  // as soon as it finishes the previous one OR as soon as dependencies are met.
  // The start time is the maximum of:
  // 1. When the unit becomes free (currentCycle)
  // 2. When all dependencies are satisfied (earliestStart)
  int64_t startTime = std::max(currentCycle, earliestStart);

  op.startCycle = startTime;
  op.endCycle = startTime + op.duration;

  // Update when this unit will be free for the next operation
  currentCycle = op.endCycle;
  scheduledOps.push_back(&op);
}

int64_t HWUnitPipeline::getTotalBusyCycles() const {
  int64_t total = 0;
  for (const auto *op : scheduledOps) {
    total += op->duration;
  }
  return total;
}

double HWUnitPipeline::getUtilization(int64_t totalCycles) const {
  if (totalCycles == 0)
    return 0.0;
  return static_cast<double>(getTotalBusyCycles()) / totalCycles * 100.0;
}

//===----------------------------------------------------------------------===//
// DependencyGraph Implementation
//===----------------------------------------------------------------------===//

void DependencyGraph::addOp(int64_t opId, Operation *op) {
  ops[opId] = op;
  if (edges.find(opId) == edges.end()) {
    edges[opId] = {};
  }
  if (reverseEdges.find(opId) == reverseEdges.end()) {
    reverseEdges[opId] = {};
  }
}

void DependencyGraph::addDependency(int64_t fromId, int64_t toId) {
  edges[fromId].push_back(toId);
  reverseEdges[toId].push_back(fromId);
}

llvm::SmallVector<int64_t, 4>
DependencyGraph::getDependencies(int64_t opId) const {
  auto it = reverseEdges.find(opId);
  if (it != reverseEdges.end()) {
    return it->second;
  }
  return {};
}

std::vector<int64_t> DependencyGraph::getTopologicalOrder() const {
  std::vector<int64_t> order;
  llvm::DenseMap<int64_t, int> inDegree;

  // Initialize in-degrees
  for (const auto &[id, _] : ops) {
    inDegree[id] = 0;
  }

  for (const auto &[id, successors] : edges) {
    for (int64_t succ : successors) {
      inDegree[succ]++;
    }
  }

  // Find all nodes with zero in-degree
  std::queue<int64_t> queue;
  for (const auto &[id, degree] : inDegree) {
    if (degree == 0) {
      queue.push(id);
    }
  }

  // Process nodes
  while (!queue.empty()) {
    int64_t node = queue.front();
    queue.pop();
    order.push_back(node);

    auto it = edges.find(node);
    if (it != edges.end()) {
      for (int64_t succ : it->second) {
        inDegree[succ]--;
        if (inDegree[succ] == 0) {
          queue.push(succ);
        }
      }
    }
  }

  return order;
}

bool DependencyGraph::hasCycle() const {
  return getTopologicalOrder().size() != ops.size();
}

//===----------------------------------------------------------------------===//
// PipelineScheduler Implementation
//===----------------------------------------------------------------------===//

PipelineScheduler::PipelineScheduler(const HardwareConfig *config)
    : hwConfig(config), ownsConfig(false), totalCycles(0) {
  if (!hwConfig) {
    // Use global config or create default
    hwConfig = static_cast<const HardwareConfig *>(&getHardwareConfig());
  }
  initPipelines();
}

void PipelineScheduler::initPipelines() {
  // Initialize pipelines for all hardware units defined in the config
  // For now, we use the standard 910B units
  pipelines.emplace(HWUnit::Cube, HWUnitPipeline(HWUnit::Cube));
  pipelines.emplace(HWUnit::CubeMTE2, HWUnitPipeline(HWUnit::CubeMTE2));
  pipelines.emplace(HWUnit::FixPipe, HWUnitPipeline(HWUnit::FixPipe));
  pipelines.emplace(HWUnit::Vector, HWUnitPipeline(HWUnit::Vector));
  pipelines.emplace(HWUnit::VecMTE2, HWUnitPipeline(HWUnit::VecMTE2));
  pipelines.emplace(HWUnit::MTE3, HWUnitPipeline(HWUnit::MTE3));
  pipelines.emplace(HWUnit::Scalar, HWUnitPipeline(HWUnit::Scalar));
}

void PipelineScheduler::addOperation(PipelineOp op) {
  depGraph.addOp(op.opId, op.mlirOp);
  operations.push_back(op);
}

void PipelineScheduler::addDependency(int64_t fromId, int64_t toId) {
  depGraph.addDependency(fromId, toId);
}

int64_t PipelineScheduler::getEarliestStartTime(const PipelineOp &op) {
  int64_t earliest = 0;

  // Check all dependencies - must wait for ALL producers to complete
  for (int64_t depId : op.dependsOn) {
    // Find the dependent operation
    for (const auto &depOp : operations) {
      if (depOp.opId == depId) {
        earliest = std::max(earliest, depOp.endCycle);
        break;
      }
    }
  }

  return earliest;
}

bool PipelineScheduler::schedule() {
  if (depGraph.hasCycle()) {
    return false;
  }

  // Get topological order - ensures we schedule producers before consumers
  std::vector<int64_t> order = depGraph.getTopologicalOrder();

  // Create a map from opId to operation index
  llvm::DenseMap<int64_t, size_t> opIdToIndex;
  for (size_t i = 0; i < operations.size(); ++i) {
    opIdToIndex[operations[i].opId] = i;
  }

  // ASAP Scheduling: Schedule each operation as early as possible
  // Constraints:
  // 1. Data dependencies (must wait for all producers)
  // 2. Resource availability (each HW unit can only run one op at a time)
  //
  // NOTE: In Ascend 910B, DIFFERENT hardware units can ALL run in parallel.
  // Only the SAME unit has serialization constraints.

  for (int64_t opId : order) {
    auto it = opIdToIndex.find(opId);
    if (it == opIdToIndex.end())
      continue;

    PipelineOp &op = operations[it->second];

    // Get earliest start time based on DATA DEPENDENCIES
    int64_t earliestStart = getEarliestStartTime(op);

    // Get the pipeline for this hardware unit
    auto pipeIt = pipelines.find(op.hwUnit);
    if (pipeIt == pipelines.end())
      continue;

    // Schedule on the appropriate pipeline
    // The HWUnitPipeline will handle resource availability constraints
    pipeIt->second.scheduleOp(op, earliestStart);

    // Update total cycles (the overall completion time)
    totalCycles = std::max(totalCycles, op.endCycle);
  }

  return true;
}

int64_t PipelineScheduler::getKernelCycles(int64_t numPrograms,
                                           int64_t numParallelUnits,
                                           int64_t numInnerIters) const {
  if (numPrograms <= 0 || numParallelUnits <= 0)
    return totalCycles;

  // Step 1: add pipe_barrier sync cost for the inner loop iterations.
  int64_t barrierCycles =
      numInnerIters * hwConfig->getPipeBarrierCyclesPerIter();

  // Step 2: apply scalar overhead factor (captures loop control, address
  // arithmetic, and AIC↔AIV synchronisation bubbles).
  double scalarFactor = hwConfig->getAIVScalarOverheadFactor();
  int64_t perProgramCycles = static_cast<int64_t>(
      (totalCycles + barrierCycles) * (1.0 + scalarFactor));

  // Step 3: multiply by number of waves (wave serialisation).
  int64_t numWaves = (numPrograms + numParallelUnits - 1) / numParallelUnits;
  return perProgramCycles * numWaves;
}

const HWUnitPipeline &PipelineScheduler::getPipeline(HWUnit unit) const {
  static HWUnitPipeline emptyPipeline(HWUnit::Scalar);
  auto it = pipelines.find(unit);
  if (it != pipelines.end()) {
    return it->second;
  }
  return emptyPipeline;
}

void PipelineScheduler::printTimeline(llvm::raw_ostream &os) const {
  // Collect all operations sorted by start time
  std::vector<const PipelineOp *> sortedOps;
  for (const auto &op : operations) {
    sortedOps.push_back(&op);
  }
  std::sort(sortedOps.begin(), sortedOps.end(),
            [](const PipelineOp *a, const PipelineOp *b) {
              return a->startCycle < b->startCycle;
            });

  // Get hardware info from config
  double freqGHz = hwConfig->getClockFrequencyGHz();
  double cyclesPerUs = freqGHz * 1000.0;

  // Print header
  os << "\n";
  os << "Hardware: " << hwConfig->getName() << " @ "
     << llvm::format("%.2f", freqGHz) << " GHz\n";
  os << "Pipeline Model: Fully pipelined (all units can run in parallel)\n\n";

  os << "┌────────┬──────────────────────┬────────────┬──────────────┬─────────"
        "─────┐\n";
  os << "│ Op ID  │ Operation            │   HW Unit  │    Start     │     End "
        "     │\n";
  os << "├────────┼──────────────────────┼────────────┼──────────────┼─────────"
        "─────┤\n";

  // Print each operation
  for (const auto *op : sortedOps) {
    std::string opName = op->opName;
    if (opName.length() > 20) {
      opName = opName.substr(0, 17) + "...";
    }
    opName.resize(20, ' ');

    std::string hwUnitStr = stringifyHWUnit(op->hwUnit).str();
    hwUnitStr.resize(10, ' ');

    os << "| " << llvm::format("%6ld", op->opId) << " | " << opName << " | "
       << hwUnitStr << " | " << llvm::format("%12ld", op->startCycle) << " | "
       << llvm::format("%12ld", op->endCycle) << " |\n";
  }

  os << "└────────┴──────────────────────┴────────────┴──────────────┴─────────"
        "─────┘\n";
  os << llvm::format("\nTotal execution time: %ld cycles (%.3f μs)\n",
                     totalCycles, totalCycles / cyclesPerUs);
}

void PipelineScheduler::printUtilizationReport(llvm::raw_ostream &os) const {
  double freqGHz = hwConfig->getClockFrequencyGHz();

  os << "\n=== Hardware Unit Utilization ===\n";
  os << "All units can execute in parallel (fully pipelined)\n\n";

  // Group by path for clarity
  os << "Cube Path (HBM -> L1 -> L0A/B -> Cube -> L0C -> HBM):\n";
  for (HWUnit unit : {HWUnit::CubeMTE2, HWUnit::Cube, HWUnit::FixPipe}) {
    auto it = pipelines.find(unit);
    if (it == pipelines.end())
      continue;

    const auto &pipeline = it->second;
    int64_t busyCycles = pipeline.getTotalBusyCycles();
    if (busyCycles > 0 || true) { // Always show
      double utilization = pipeline.getUtilization(totalCycles);
      std::string unitStr = stringifyHWUnit(unit).str();
      unitStr.resize(12, ' ');

      int barWidth = static_cast<int>(utilization / 5);
      std::string bar(barWidth, '#');
      bar.resize(20, '-');

      os << "  " << unitStr << " [" << bar << "] "
         << llvm::format("%6.2f", utilization) << "% (" << busyCycles
         << " cycles)\n";
    }
  }

  os << "\nVector Path (HBM -> UB -> Vector -> UB -> HBM):\n";
  for (HWUnit unit : {HWUnit::VecMTE2, HWUnit::Vector, HWUnit::MTE3}) {
    auto it = pipelines.find(unit);
    if (it == pipelines.end())
      continue;

    const auto &pipeline = it->second;
    int64_t busyCycles = pipeline.getTotalBusyCycles();
    double utilization = pipeline.getUtilization(totalCycles);
    std::string unitStr = stringifyHWUnit(unit).str();
    unitStr.resize(12, ' ');

    int barWidth = static_cast<int>(utilization / 5);
    std::string bar(barWidth, '#');
    bar.resize(20, '-');

    os << "  " << unitStr << " [" << bar << "] "
       << llvm::format("%6.2f", utilization) << "% (" << busyCycles
       << " cycles)\n";
  }

  // Find bottleneck (unit with highest utilization)
  double maxUtil = 0;
  HWUnit bottleneck = HWUnit::Scalar;
  for (const auto &[unit, pipeline] : pipelines) {
    double util = pipeline.getUtilization(totalCycles);
    if (util > maxUtil) {
      maxUtil = util;
      bottleneck = unit;
    }
  }

  if (maxUtil > 0) {
    os << "\nBottleneck: " << stringifyHWUnit(bottleneck).str()
       << llvm::format(" (%.2f%% utilization)\n", maxUtil);
  }

  os << "\n";
}

//===----------------------------------------------------------------------===//
// PerformanceReport Implementation
//===----------------------------------------------------------------------===//

void PerformanceReport::print(llvm::raw_ostream &os) const {
  os << "\n";
  os << "╔══════════════════════════════════════════════════════════════╗\n";
  os << "║          " << hardwareName << " Performance Analysis Report";
  // Pad to align
  int padding = 50 - hardwareName.length();
  for (int i = 0; i < padding; ++i)
    os << " ";
  os << "║\n";
  os << "╠══════════════════════════════════════════════════════════════╣\n";

  // Timing
  os << "║  Timing Summary                                              ║\n";
  os << llvm::format(
      "║    Clock Frequency:       %8.2f GHz                       ║\n",
      clockFreqGHz);
  os << llvm::format(
      "║    Total Cycles (compute):%12ld                       ║\n",
      totalCycles);
  os << llvm::format(
      "║    Estimated Time:        %12.3f μs                    ║\n",
      totalTimeUs);
  if (kernelTotalCycles > 0) {
    double kernelTimeUs = kernelTotalCycles / (clockFreqGHz * 1000.0);
    os << llvm::format(
        "║    Kernel Cycles (+ovhd): %12ld                       ║\n",
        kernelTotalCycles);
    os << llvm::format(
        "║    Kernel Time (+ovhd):   %12.3f μs                    ║\n",
        kernelTimeUs);
    os << llvm::format(
        "║    Num Waves:             %12ld                       ║\n",
        numWaves);
  }

  os << "║                                                              ║\n";

  // Roofline
  os << "║  Roofline Analysis                                           ║\n";
  os << llvm::format(
      "║    Arithmetic Intensity:  %8.3f FLOP/Byte                ║\n",
      arithmeticIntensity);
  os << llvm::format(
      "║    Achieved TFLOPS:       %8.3f                          ║\n",
      achievedTFLOPS);
  os << llvm::format(
      "║    Peak TFLOPS:           %8.3f                          ║\n",
      peakTFLOPS);
  os << llvm::format(
      "║    Achieved Bandwidth:    %8.1f GB/s                     ║\n",
      achievedBandwidth);
  os << llvm::format(
      "║    Peak Bandwidth:        %8.1f GB/s                     ║\n",
      peakBandwidth);
  os << "║    Bound:                 "
     << (isComputeBound ? "Compute-bound" : "Memory-bound ")
     << "                      ║\n";

  os << "║                                                              ║\n";

  // Utilization
  os << "║  Hardware Unit Utilization                                   ║\n";
  for (const auto &[unit, util] : unitUtilization) {
    if (util > 0.01) { // Only show if > 0.01%
      std::string unitStr = stringifyHWUnit(unit).str();
      unitStr.resize(12, ' ');

      int barWidth = static_cast<int>(util / 5);
      std::string bar(barWidth, '#');
      bar.resize(20, '-');

      os << "║    " << unitStr << " [" << bar << "] "
         << llvm::format("%5.1f%%", util) << "         ║\n";
    }
  }

  os << "╚══════════════════════════════════════════════════════════════╝\n";
}

std::string PerformanceReport::toJSON() const {
  std::string json = "{\n";
  json += "  \"hardware\": \"" + hardwareName + "\",\n";
  json += "  \"clock_freq_ghz\": " + std::to_string(clockFreqGHz) + ",\n";
  json += "  \"total_cycles\": " + std::to_string(totalCycles) + ",\n";
  json += "  \"total_time_us\": " + std::to_string(totalTimeUs) + ",\n";
  json += "  \"arithmetic_intensity\": " + std::to_string(arithmeticIntensity) +
          ",\n";
  json += "  \"achieved_tflops\": " + std::to_string(achievedTFLOPS) + ",\n";
  json += "  \"peak_tflops\": " + std::to_string(peakTFLOPS) + ",\n";
  json +=
      "  \"achieved_bandwidth_gbps\": " + std::to_string(achievedBandwidth) +
      ",\n";
  json += "  \"peak_bandwidth_gbps\": " + std::to_string(peakBandwidth) + ",\n";
  json += "  \"is_compute_bound\": " +
          std::string(isComputeBound ? "true" : "false") + ",\n";
  json += "  \"bottleneck_unit\": \"" + stringifyHWUnit(bottleneckUnit).str() +
          "\",\n";
  json +=
      "  \"bottleneck_utilization\": " + std::to_string(bottleneckUtilization) +
      ",\n";

  json += "  \"unit_utilization\": {\n";
  bool first = true;
  for (const auto &[unit, util] : unitUtilization) {
    if (!first)
      json += ",\n";
    first = false;
    json +=
        "    \"" + stringifyHWUnit(unit).str() + "\": " + std::to_string(util);
  }
  json += "\n  }\n";
  json += "}\n";

  return json;
}

//===----------------------------------------------------------------------===//
// RooflineAnalyzer Implementation
//===----------------------------------------------------------------------===//

RooflineAnalyzer::RooflineAnalyzer(const PipelineScheduler &sched)
    : scheduler(sched), config(sched.getConfig()), totalFLOPs(0),
      totalBytes(0) {
  computeMetrics();
}

void RooflineAnalyzer::computeMetrics() {
  totalFLOPs = 0;
  totalBytes = 0;

  for (const auto &op : scheduler.getAllOps()) {
    totalFLOPs += op.flops;
    totalBytes += op.bytes;
  }
}

double RooflineAnalyzer::getArithmeticIntensity() const {
  if (totalBytes == 0)
    return 0.0;
  return static_cast<double>(totalFLOPs) / totalBytes;
}

double RooflineAnalyzer::getTheoreticalPeak() const {
  // Peak compute: Cube FP16 TFLOPS
  return config.getCubeTFlopsFP16();
}

double RooflineAnalyzer::getAchievedPerformance() const {
  int64_t cycles = scheduler.getTotalCycles();
  if (cycles == 0)
    return 0.0;

  double freqGHz = config.getClockFrequencyGHz();
  double timeSeconds = cycles / (freqGHz * 1e9);

  return (totalFLOPs / 1e12) / timeSeconds; // TFLOPS
}

bool RooflineAnalyzer::isComputeBound() const {
  double ai = getArithmeticIntensity();
  double peakTFLOPS = getTheoreticalPeak();
  double peakBWTBps = config.getMemoryBandwidthTBps("hbm");

  // Ridge point: where compute ceiling meets memory ceiling
  // ridge_ai = peak_compute / peak_bandwidth
  double ridgePoint = peakTFLOPS / peakBWTBps; // FLOP/Byte

  return ai >= ridgePoint;
}

PerformanceReport RooflineAnalyzer::analyze() {
  PerformanceReport report;

  // Hardware info
  report.hardwareName = config.getName().str();
  report.clockFreqGHz = config.getClockFrequencyGHz();

  // Timing
  report.totalCycles = scheduler.getTotalCycles();
  report.totalTimeUs = config.cyclesToMicroseconds(report.totalCycles);

  // Roofline
  report.arithmeticIntensity = getArithmeticIntensity();
  report.achievedTFLOPS = getAchievedPerformance();
  report.peakTFLOPS = getTheoreticalPeak();
  report.isComputeBound = isComputeBound();

  // Bandwidth
  report.peakBandwidth = config.getMemoryBandwidthTBps("hbm") * 1000.0; // GB/s
  if (report.totalCycles > 0) {
    double timeSeconds = report.totalCycles / (report.clockFreqGHz * 1e9);
    report.achievedBandwidth = (totalBytes / 1e9) / timeSeconds; // GB/s
  } else {
    report.achievedBandwidth = 0;
  }

  // Utilization
  double maxUtil = 0;
  for (HWUnit unit : {HWUnit::Cube, HWUnit::CubeMTE2, HWUnit::FixPipe,
                      HWUnit::Vector, HWUnit::VecMTE2, HWUnit::MTE3}) {
    const auto &pipeline = scheduler.getPipeline(unit);
    double util = pipeline.getUtilization(report.totalCycles);
    report.unitUtilization[unit] = util;
    report.unitBusyCycles[unit] = pipeline.getTotalBusyCycles();

    if (util > maxUtil) {
      maxUtil = util;
      report.bottleneckUnit = unit;
    }
  }
  report.bottleneckUtilization = maxUtil;

  return report;
}
