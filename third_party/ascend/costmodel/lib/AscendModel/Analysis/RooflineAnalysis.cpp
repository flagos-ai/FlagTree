//===- RooflineAnalysis.cpp - Roofline model analysis ---------------------===//
//
// This file implements the roofline model analysis for Ascend 910B.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareParams.h"
#include "AscendModel/IR/AscendModelDialect.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::ascend;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static int64_t getNumElementsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 1024;
      count *= dim;
    }
    return count;
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// RooflineAnalyzer Implementation
//===----------------------------------------------------------------------===//

RooflineAnalyzer::RooflineAnalyzer(const PipelineScheduler &scheduler)
    : scheduler(scheduler), totalFLOPs(0), totalBytes(0) {
  computeMetrics();
}

void RooflineAnalyzer::computeMetrics() {
  for (const auto &op : scheduler.getAllOps()) {
    if (!op.mlirOp)
      continue;

    Operation *mlirOp = op.mlirOp;

    // Count FLOPs based on operation type
    if (auto matmulOp = dyn_cast<MatmulOp>(mlirOp)) {
      int64_t m = matmulOp.getM();
      int64_t n = matmulOp.getN();
      int64_t k = matmulOp.getK();
      totalFLOPs += 2 * m * n * k;
    }
    // Binary ops - 1 FLOP per element
    else if (isa<AddOp, SubOp, MulOp, MaxOp, MinOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs += getNumElementsFromType(mlirOp->getResult(0).getType());
      }
    }
    // Division - more expensive
    else if (isa<DivOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs +=
            getNumElementsFromType(mlirOp->getResult(0).getType()) * 10;
      }
    }
    // Simple unary ops
    else if (isa<NegOp, AbsOp, ReluOp, CastOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs += getNumElementsFromType(mlirOp->getResult(0).getType());
      }
    }
    // Transcendental functions — differentiated FLOP costs
    // Reflects relative complexity on Ascend 910B Vector unit
    else if (isa<ExpOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs +=
            getNumElementsFromType(mlirOp->getResult(0).getType()) * 15;
      }
    } else if (isa<LogOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs +=
            getNumElementsFromType(mlirOp->getResult(0).getType()) * 20;
      }
    } else if (isa<TanhOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs +=
            getNumElementsFromType(mlirOp->getResult(0).getType()) * 30;
      }
    } else if (isa<SigmoidOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs +=
            getNumElementsFromType(mlirOp->getResult(0).getType()) * 25;
      }
    }
    // Sqrt operations
    else if (isa<SqrtOp, RsqrtOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs +=
            getNumElementsFromType(mlirOp->getResult(0).getType()) * 10;
      }
    }
    // Reduction ops
    else if (isa<ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp>(mlirOp)) {
      if (mlirOp->getNumOperands() > 0) {
        totalFLOPs += getNumElementsFromType(mlirOp->getOperand(0).getType());
      }
    }
    // Select/where op — 1 FLOP per element (comparison + select)
    else if (isa<SelectOp>(mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs += getNumElementsFromType(mlirOp->getResult(0).getType());
      }
    }
    // Comparison ops — 1 FLOP per element
    else if (isa<CmpEqOp, CmpNeOp, CmpLtOp, CmpLeOp, CmpGtOp, CmpGeOp>(
                 mlirOp)) {
      if (mlirOp->getNumResults() > 0) {
        totalFLOPs += getNumElementsFromType(mlirOp->getResult(0).getType());
      }
    }

    // Count bytes for memory operations
    if (auto loadOp = dyn_cast<CubeLoadOp>(mlirOp)) {
      totalBytes += loadOp.getBytes();
    } else if (auto storeOp = dyn_cast<CubeStoreOp>(mlirOp)) {
      totalBytes += storeOp.getBytes();
    } else if (auto loadOp = dyn_cast<VectorLoadOp>(mlirOp)) {
      totalBytes += loadOp.getBytes();
    } else if (auto storeOp = dyn_cast<VectorStoreOp>(mlirOp)) {
      totalBytes += storeOp.getBytes();
    }
  }
}

double RooflineAnalyzer::getArithmeticIntensity() const {
  if (totalBytes == 0)
    return 0.0;
  return static_cast<double>(totalFLOPs) / totalBytes;
}

bool RooflineAnalyzer::isComputeBound() const {
  double ai = getArithmeticIntensity();
  double ridgePoint =
      ::ascend::hw::CUBE_TFLOPS / ::ascend::hw::HBM_BANDWIDTH_GBS;
  return ai > ridgePoint;
}

double RooflineAnalyzer::getTheoreticalPeak() const {
  double ai = getArithmeticIntensity();
  double memoryRoof = ::ascend::hw::HBM_BANDWIDTH_GBS * ai;
  return std::min(::ascend::hw::CUBE_TFLOPS, memoryRoof);
}

double RooflineAnalyzer::getAchievedPerformance() const {
  int64_t totalCycles = scheduler.getTotalCycles();
  if (totalCycles == 0)
    return 0.0;

  double timeSeconds = totalCycles / (::ascend::hw::CORE_FREQUENCY_GHZ * 1e9);
  return (totalFLOPs / 1e12) / timeSeconds;
}

PerformanceReport RooflineAnalyzer::analyze() {
  PerformanceReport report;

  report.totalCycles = scheduler.getTotalCycles();
  report.totalTimeUs = report.totalCycles / ::ascend::hw::CYCLES_PER_US;

  for (int i = 0; i <= static_cast<int>(HWUnit::Scalar); ++i) {
    HWUnit unit = static_cast<HWUnit>(i);
    const auto &pipeline = scheduler.getPipeline(unit);
    report.unitBusyCycles[unit] = pipeline.getTotalBusyCycles();
    report.unitUtilization[unit] = pipeline.getUtilization(report.totalCycles);
  }

  report.bottleneckUtilization = 0;
  report.bottleneckUnit = HWUnit::Scalar;
  for (const auto &[unit, util] : report.unitUtilization) {
    if (util > report.bottleneckUtilization) {
      report.bottleneckUtilization = util;
      report.bottleneckUnit = unit;
    }
  }

  report.arithmeticIntensity = getArithmeticIntensity();
  report.achievedTFLOPS = getAchievedPerformance();
  report.peakTFLOPS = ::ascend::hw::CUBE_TFLOPS;
  report.achievedBandwidth =
      totalBytes > 0 ? (totalBytes / 1e9) / (report.totalTimeUs / 1e6) : 0;
  report.peakBandwidth = ::ascend::hw::HBM_BANDWIDTH_GBS;
  report.isComputeBound = isComputeBound();

  std::map<std::string, int64_t> opCounts;
  std::map<std::string, int64_t> opCycles;

  for (const auto &op : scheduler.getAllOps()) {
    opCounts[op.opName]++;
    opCycles[op.opName] += op.duration;
  }

  for (const auto &[name, count] : opCounts) {
    PerformanceReport::OpStat stat;
    stat.opType = name;
    stat.count = count;
    stat.totalCycles = opCycles[name];
    stat.percentage = report.totalCycles > 0
                          ? 100.0 * opCycles[name] / report.totalCycles
                          : 0;
    report.opStats.push_back(stat);
  }

  std::sort(
      report.opStats.begin(), report.opStats.end(),
      [](const auto &a, const auto &b) { return a.percentage > b.percentage; });

  return report;
}

//===----------------------------------------------------------------------===//
// PerformanceReport Implementation
//===----------------------------------------------------------------------===//

void PerformanceReport::print(llvm::raw_ostream &os) const {
  os << "\n";
  os << "+====================================================================="
        "=+\n";
  os << "|               Ascend 910B Roofline Performance Report               "
        "|\n";
  os << "+====================================================================="
        "=+\n";
  os << "|                                                                     "
        " |\n";
  os << "|  Timing                                                             "
        " |\n";
  os << "|  ------                                                             "
        " |\n";
  os << llvm::format(
      "|    Total Cycles:          %15ld                          |\n",
      totalCycles);
  os << llvm::format(
      "|    Estimated Time:        %15.3f us                        |\n",
      totalTimeUs);
  os << "|                                                                     "
        " |\n";
  os << "|  Roofline Analysis                                                  "
        " |\n";
  os << "|  -----------------                                                  "
        " |\n";
  os << llvm::format(
      "|    Arithmetic Intensity:  %15.3f FLOP/Byte               |\n",
      arithmeticIntensity);
  os << llvm::format(
      "|    Achieved TFLOPS:       %15.3f                          |\n",
      achievedTFLOPS);
  os << llvm::format(
      "|    Peak TFLOPS:           %15.1f                          |\n",
      peakTFLOPS);
  os << llvm::format(
      "|    Achieved Bandwidth:    %15.3f GB/s                     |\n",
      achievedBandwidth);
  os << llvm::format(
      "|    Peak Bandwidth:        %15.1f GB/s                     |\n",
      peakBandwidth);
  os << "|    Bound:                 ";
  os << (isComputeBound ? "   Compute-bound" : "    Memory-bound");
  os << "                            |\n";
  os << "|                                                                     "
        " |\n";
  os << "|  Bottleneck                                                         "
        " |\n";
  os << "|  ----------                                                         "
        " |\n";
  os << "|    Unit:                  ";
  os << llvm::format("%-15s", stringifyHWUnit(bottleneckUnit).str().c_str());
  os << "                         |\n";
  os << llvm::format(
      "|    Utilization:           %14.2f%%                         |\n",
      bottleneckUtilization);
  os << "|                                                                     "
        " |\n";
  os << "+====================================================================="
        "=+\n";
}

std::string PerformanceReport::toJSON() const {
  llvm::json::Object root;

  root["total_cycles"] = totalCycles;
  root["total_time_us"] = totalTimeUs;

  llvm::json::Object utilization;
  for (const auto &[unit, util] : unitUtilization) {
    utilization[stringifyHWUnit(unit).str()] = util;
  }
  root["unit_utilization"] = std::move(utilization);

  llvm::json::Object roofline;
  roofline["arithmetic_intensity"] = arithmeticIntensity;
  roofline["achieved_tflops"] = achievedTFLOPS;
  roofline["peak_tflops"] = peakTFLOPS;
  roofline["achieved_bandwidth_gbs"] = achievedBandwidth;
  roofline["peak_bandwidth_gbs"] = peakBandwidth;
  roofline["is_compute_bound"] = isComputeBound;
  root["roofline"] = std::move(roofline);

  llvm::json::Object bottleneck;
  bottleneck["unit"] = stringifyHWUnit(bottleneckUnit).str();
  bottleneck["utilization"] = bottleneckUtilization;
  root["bottleneck"] = std::move(bottleneck);

  llvm::json::Array opStatsArray;
  for (const auto &stat : opStats) {
    llvm::json::Object statObj;
    statObj["op_type"] = stat.opType;
    statObj["count"] = stat.count;
    statObj["total_cycles"] = stat.totalCycles;
    statObj["percentage"] = stat.percentage;
    opStatsArray.push_back(std::move(statObj));
  }
  root["op_stats"] = std::move(opStatsArray);

  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::json::Value(std::move(root));
  return result;
}
