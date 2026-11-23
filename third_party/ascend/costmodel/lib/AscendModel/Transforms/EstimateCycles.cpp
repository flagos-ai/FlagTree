//===- EstimateCycles.cpp - Estimate execution cycles --------------------===//
//
// This file estimates execution cycles for AscendModel operations using
// the Roofline model. Compute and memory operations can overlap, so we
// take max(compute_time, memory_time) instead of sum.
//
// For Triton kernels with loops depending on tt.get_program_id, use:
//   inproc-costmodel --estimate-cycles="arg-bindings=arg2=100,pid_x=0"
//   input.mlir
//
//===----------------------------------------------------------------------===//

#include "AscendModel/HardwareConfig.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/IR/AscendModelInterfaces.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_ESTIMATECYCLESPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using utils::analyzeScfForTripCount;
using utils::getLoopMultiplier;
using utils::getNumElements;
using utils::getScfForTripCount;
using utils::getScfForTripCountWithBindings;
using utils::LoopTripCountResult;
using utils::parseBindings;
using utils::parseLoopTripCounts;

//===----------------------------------------------------------------------===//
// Roofline Analysis
//===----------------------------------------------------------------------===//

/// Statistics for roofline analysis, separated by hardware path.
struct RooflineStats {
  // Cube path
  int64_t cubeFlops = 0;
  int64_t cubeLoadBytes = 0;  // CubeMTE2
  int64_t cubeStoreBytes = 0; // FixPipe

  // Vector path
  int64_t vectorFlops = 0;
  int64_t vectorLoadBytes = 0;  // VecMTE2
  int64_t vectorStoreBytes = 0; // MTE3

  // Cycle estimates (per-op, for annotation)
  int64_t cubeCycles = 0;
  int64_t cubeLoadCycles = 0;
  int64_t cubeStoreCycles = 0;
  int64_t vectorCycles = 0;
  int64_t vectorLoadCycles = 0;
  int64_t vectorStoreCycles = 0;

  /// Calculate total cycles using roofline model.
  /// For each path (Cube/Vector), take max of compute and memory time.
  /// Then consider if paths can overlap or must serialize.
  int64_t calculateRooflineCycles(const HardwareConfig &config,
                                  bool cubeVectorOverlap = true) const {
    // Cube path: max(compute, load, store)
    int64_t cubePathCycles =
        std::max({cubeCycles, cubeLoadCycles, cubeStoreCycles});

    // Vector path: max(compute, load, store)
    int64_t vectorPathCycles =
        std::max({vectorCycles, vectorLoadCycles, vectorStoreCycles});

    // If Cube and Vector can overlap, take max; otherwise sum
    if (cubeVectorOverlap) {
      return std::max(cubePathCycles, vectorPathCycles);
    } else {
      return cubePathCycles + vectorPathCycles;
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct EstimateCyclesPass
    : public impl::EstimateCyclesPassBase<EstimateCyclesPass> {
  using EstimateCyclesPassBase::EstimateCyclesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::string hardwareConfigError;
    auto hardwareConfig =
        loadHardwareConfigForAnalysis(hardwareConfigPath, hardwareConfigError);
    if (!hardwareConfig) {
      emitError(module.getLoc(), hardwareConfigError);
      return signalPassFailure();
    }
    const HardwareConfig &config = *hardwareConfig;

    // Parse bindings
    llvm::DenseMap<unsigned, int64_t> argBindings;
    llvm::StringMap<int64_t> programIdBindings;
    if (!argBindingsStr.empty()) {
      std::string parseError;
      if (!parseBindings(argBindingsStr, argBindings, programIdBindings,
                         parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }

    // Parse loop trip count overrides
    SmallVector<int64_t> loopTripCountOverrides;
    if (!loopTripCountsStr.empty()) {
      std::string parseError;
      if (!parseLoopTripCounts(loopTripCountsStr, loopTripCountOverrides,
                               parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }

    bool hasError = false;

    // First pass: collect all loops and assign trip counts
    SmallVector<scf::ForOp> allLoops;
    module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });

    for (size_t loopIdx = 0; loopIdx < allLoops.size(); ++loopIdx) {
      scf::ForOp forOp = allLoops[loopIdx];
      int64_t tripCount = 1;
      std::string source = "unknown";

      if (loopIdx < loopTripCountOverrides.size()) {
        tripCount = loopTripCountOverrides[loopIdx];
        source = "override";
      } else {
        auto result = getScfForTripCountWithBindings(forOp, argBindings,
                                                     programIdBindings);

        if (result.isStatic) {
          tripCount = result.staticTripCount;
          source = result.dependentArgs.empty() &&
                           result.lowerBound.dependentProgramIds.empty() &&
                           result.upperBound.dependentProgramIds.empty()
                       ? "static"
                       : "evaluated";
        } else if (!result.errorMsg.empty()) {
          std::string errorMsg = "Loop " + std::to_string(loopIdx) +
                                 " trip count cannot be determined.\n";
          errorMsg += "  Lower bound: " + result.lowerBound.description + "\n";
          errorMsg += "  Upper bound: " + result.upperBound.description + "\n";
          errorMsg += "  Step: " + result.step.description + "\n";
          errorMsg += result.errorMsg;

          emitError(forOp.getLoc(), errorMsg);
          hasError = true;
          continue;
        }
      }

      forOp->setAttr("ascend.trip_count",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64),
                                      tripCount));
      forOp->setAttr("ascend.trip_count_source",
                     StringAttr::get(forOp.getContext(), source));
    }

    if (hasError)
      return signalPassFailure();

    // Second pass: estimate cycles for each operation and collect roofline
    // stats
    RooflineStats stats;
    int64_t totalOps = 0;

    module.walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::YieldOp, scf::IfOp>(op))
        return;

      if (auto cyclesOp = dyn_cast<EstimateCyclesOpInterface>(op)) {
        int64_t cycles = cyclesOp.estimateCycles(config);
        if (cycles <= 0)
          return;

        HWUnit hwUnit = cyclesOp.getHWUnit();
        int64_t loopMultiplier = getLoopMultiplier(op);
        int64_t totalOpCycles = cycles * loopMultiplier;

        // Set attributes on the operation
        op->setAttr(
            "estimated_cycles",
            IntegerAttr::get(IntegerType::get(op->getContext(), 64), cycles));
        op->setAttr("hw_unit",
                    StringAttr::get(op->getContext(), stringifyHWUnit(hwUnit)));

        if (int64_t bytes = cyclesOp.getTransferBytes(); bytes > 0) {
          op->setAttr(
              "bytes",
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), bytes));
        }
        if (int64_t flops = cyclesOp.getFlops(); flops > 0) {
          op->setAttr(
              "flops",
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), flops));
        }
        if (loopMultiplier > 1) {
          op->setAttr("loop_multiplier",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                       loopMultiplier));
        }

        totalOps++;

        // Accumulate roofline stats by hardware unit
        int64_t bytes = cyclesOp.getTransferBytes() * loopMultiplier;
        int64_t flops = cyclesOp.getFlops() * loopMultiplier;

        switch (hwUnit) {
        case HWUnit::Cube:
          stats.cubeFlops += flops;
          stats.cubeCycles += totalOpCycles;
          break;
        case HWUnit::CubeMTE2:
          stats.cubeLoadBytes += bytes;
          stats.cubeLoadCycles += totalOpCycles;
          break;
        case HWUnit::FixPipe:
          stats.cubeStoreBytes += bytes;
          stats.cubeStoreCycles += totalOpCycles;
          break;
        case HWUnit::Vector:
          stats.vectorFlops += flops;
          stats.vectorCycles += totalOpCycles;
          break;
        case HWUnit::VecMTE2:
          stats.vectorLoadBytes += bytes;
          stats.vectorLoadCycles += totalOpCycles;
          break;
        case HWUnit::MTE3:
          stats.vectorStoreBytes += bytes;
          stats.vectorStoreCycles += totalOpCycles;
          break;
        default:
          break;
        }
      }
    });

    // Calculate roofline-based total cycles
    int64_t rooflineCycles = stats.calculateRooflineCycles(config, true);

    // Also calculate simple sum for comparison
    int64_t simpleSumCycles = stats.cubeCycles + stats.cubeLoadCycles +
                              stats.cubeStoreCycles + stats.vectorCycles +
                              stats.vectorLoadCycles + stats.vectorStoreCycles;

    // Third pass: annotate loops with body cycles
    module.walk([&](scf::ForOp forOp) {
      int64_t tripCount = 1;
      if (auto attr = forOp->getAttrOfType<IntegerAttr>("ascend.trip_count"))
        tripCount = attr.getInt();

      int64_t bodyCycles = 0;
      int64_t bodyOps = 0;
      for (Operation &op : forOp.getBody()->getOperations()) {
        if (isa<scf::YieldOp>(&op))
          continue;
        if (auto nestedFor = dyn_cast<scf::ForOp>(&op)) {
          if (auto nestedTotal =
                  nestedFor->getAttrOfType<IntegerAttr>("ascend.total_cycles"))
            bodyCycles += nestedTotal.getInt();
        } else if (auto cyclesAttr =
                       op.getAttrOfType<IntegerAttr>("estimated_cycles")) {
          bodyCycles += cyclesAttr.getInt();
          bodyOps++;
        }
      }

      forOp->setAttr("ascend.body_cycles",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64),
                                      bodyCycles));
      forOp->setAttr("ascend.total_cycles",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64),
                                      bodyCycles * tripCount));
      forOp->setAttr(
          "ascend.body_ops",
          IntegerAttr::get(IntegerType::get(forOp.getContext(), 64), bodyOps));
    });

    // Set module-level attributes
    module->setAttr("ascend.roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     rooflineCycles));
    module->setAttr("ascend.simple_sum_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     simpleSumCycles));
    module->setAttr(
        "ascend.total_ops",
        IntegerAttr::get(IntegerType::get(module.getContext(), 64), totalOps));
    module->setAttr("ascend.hardware",
                    StringAttr::get(module.getContext(), config.getName()));
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
