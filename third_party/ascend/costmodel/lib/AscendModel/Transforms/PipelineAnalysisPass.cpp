//===- PipelineAnalysisPass.cpp - Pipeline scheduling analysis -----------===//
//
// Uses HardwareConfig for configurable hardware parameters.
// Handles dynamic loop bounds via arg-bindings option (supports program_id).
// Uses Roofline model for cycle estimation with HW unit overlap.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_PIPELINEANALYSISPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using utils::getLoopMultiplier;
using utils::getScfForTripCount;
using utils::getScfForTripCountWithBindings;
using utils::parseBindings;
using utils::parseLoopTripCounts;

HWUnit getOpHWUnit(Operation *op) {
  if (isa<MatmulOp>(op))
    return HWUnit::Cube;
  if (isa<CubeLoadOp>(op))
    return HWUnit::CubeMTE2;
  if (isa<CubeStoreOp>(op))
    return HWUnit::FixPipe;
  if (isa<VectorLoadOp>(op))
    return HWUnit::VecMTE2;
  if (isa<VectorStoreOp>(op))
    return HWUnit::MTE3;
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp, ExpOp, LogOp, SqrtOp,
          RsqrtOp, TanhOp, SigmoidOp, NegOp, AbsOp, ReluOp, CastOp, ReduceSumOp,
          ReduceMaxOp, ReduceMinOp, ReduceProdOp, BroadcastOp, SelectOp>(op))
    return HWUnit::Vector;
  return HWUnit::Scalar;
}

struct PipelineAnalysisPass
    : public impl::PipelineAnalysisPassBase<PipelineAnalysisPass> {
  using PipelineAnalysisPassBase::PipelineAnalysisPassBase;

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
    SmallVector<int64_t> loopTripCountOverrides;

    if (!argBindingsStr.empty()) {
      std::string parseError;
      if (!parseBindings(argBindingsStr, argBindings, programIdBindings,
                         parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }

    if (!loopTripCountsStr.empty()) {
      std::string parseError;
      if (!parseLoopTripCounts(loopTripCountsStr, loopTripCountOverrides,
                               parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }

    // Collect loops and ensure trip counts are set
    SmallVector<scf::ForOp> allLoops;
    module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });

    bool hasError = false;
    for (size_t loopIdx = 0; loopIdx < allLoops.size(); ++loopIdx) {
      scf::ForOp forOp = allLoops[loopIdx];

      if (forOp->hasAttr("ascend.trip_count"))
        continue;

      int64_t tripCount = 1;
      if (loopIdx < loopTripCountOverrides.size()) {
        tripCount = loopTripCountOverrides[loopIdx];
      } else {
        auto result = getScfForTripCountWithBindings(forOp, argBindings,
                                                     programIdBindings);
        if (result.isStatic) {
          tripCount = result.staticTripCount;
        } else {
          emitError(forOp.getLoc(), "Loop " + std::to_string(loopIdx) +
                                        " trip count unknown. " +
                                        result.errorMsg);
          hasError = true;
          continue;
        }
      }

      forOp->setAttr("ascend.trip_count",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64),
                                      tripCount));
    }

    if (hasError)
      return signalPassFailure();

    // Build scheduler
    PipelineScheduler scheduler(&config);
    llvm::DenseMap<Value, int64_t> valueProducers;

    module.walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::YieldOp, scf::IfOp>(op))
        return;

      auto opIdAttr = op->getAttrOfType<IntegerAttr>("op_id");
      if (!opIdAttr)
        return;

      int64_t opId = opIdAttr.getInt();
      auto cyclesAttr = op->getAttrOfType<IntegerAttr>("estimated_cycles");
      int64_t cycles = cyclesAttr ? cyclesAttr.getInt() : 1;

      PipelineOp pipelineOp;
      pipelineOp.opId = opId;
      pipelineOp.hwUnit = getOpHWUnit(op);
      pipelineOp.duration = cycles;
      pipelineOp.mlirOp = op;
      pipelineOp.opName = op->getName().getStringRef().str();
      pipelineOp.loopMultiplier = getLoopMultiplier(op);

      for (Value operand : op->getOperands()) {
        auto it = valueProducers.find(operand);
        if (it != valueProducers.end()) {
          pipelineOp.dependsOn.push_back(it->second);
          scheduler.addDependency(it->second, opId);
        }
      }

      for (Value result : op->getResults())
        valueProducers[result] = opId;

      scheduler.addOperation(pipelineOp);
    });

    if (!scheduler.schedule()) {
      emitError(module.getLoc(), "Failed to schedule pipeline");
      return signalPassFailure();
    }

    // Calculate cycles using roofline model
    // oneIterCycles from scheduler already considers HW unit parallelism for
    // one iteration
    int64_t oneIterCycles = scheduler.getTotalCycles();

    // For total cycles with loops, we need to consider:
    // 1. Each HW unit's total work across all iterations
    // 2. Take max (not sum) since they can overlap

    // Collect per-HW-unit cycles
    llvm::DenseMap<HWUnit, int64_t> hwUnitCycles;
    for (const auto &pipelineOp : scheduler.getAllOps()) {
      hwUnitCycles[pipelineOp.hwUnit] +=
          pipelineOp.duration * pipelineOp.loopMultiplier;
    }

    // Group by path and apply roofline model.
    // Cube path: max(Cube, CubeMTE2, FixPipe). No cube-side mutex on 910B
    // (tilesim pipe_exclusive_config only pairs AIV MTE2<->MTE3).
    int64_t cubePathCycles =
        std::max({hwUnitCycles[HWUnit::Cube], hwUnitCycles[HWUnit::CubeMTE2],
                  hwUnitCycles[HWUnit::FixPipe]});

    // Vector path. Vector compute overlaps with load/store transfers, but on
    // 910B AIV the MTE2 (load) and MTE3 (store) units share one physical
    // pipeline (tilesim MutexComponents) and must serialize. With the mutex
    // the transfer time is VecMTE2 + MTE3; without it (legacy) they are
    // assumed to overlap (max). This is root cause 3.
    int64_t vecTransfer;
    if (config.areMutexUnits("vec_mte2", "mte3"))
      vecTransfer = hwUnitCycles[HWUnit::VecMTE2] + hwUnitCycles[HWUnit::MTE3];
    else
      vecTransfer =
          std::max(hwUnitCycles[HWUnit::VecMTE2], hwUnitCycles[HWUnit::MTE3]);
    int64_t vectorPathCycles =
        std::max(hwUnitCycles[HWUnit::Vector], vecTransfer);

    // Total: max of paths (Cube and Vector paths overlap)
    int64_t rooflineTotalCycles = std::max(cubePathCycles, vectorPathCycles);

    // Also calculate simple sum for comparison
    int64_t simpleSumCycles = 0;
    for (const auto &pipelineOp : scheduler.getAllOps())
      simpleSumCycles += pipelineOp.duration * pipelineOp.loopMultiplier;

    module->setAttr("ascend.scheduled_cycles_one_iter",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     oneIterCycles));
    module->setAttr("ascend.roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     rooflineTotalCycles));
    // Public summary consumed by the in-process Python costmodel bridge.
    // This must include loop multipliers; otherwise configs with many inner
    // loop iterations are under-estimated and can incorrectly outrank faster
    // configs during costmodel prefiltering.
    module->setAttr("ascend.scheduled_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     rooflineTotalCycles));
    module->setAttr("ascend.simple_sum_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     simpleSumCycles));
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
