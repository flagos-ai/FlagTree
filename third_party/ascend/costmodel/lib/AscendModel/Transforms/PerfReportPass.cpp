//===- PerfReportPass.cpp - Generate performance report ------------------===//
//
// This file generates a comprehensive performance report.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareParams.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_PERFREPORTPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

/// Get the HWUnit for an operation
static HWUnit getOpHWUnit(Operation *op) {
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

struct PerfReportPass : public impl::PerfReportPassBase<PerfReportPass> {
  using PerfReportPassBase::PerfReportPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int64_t totalCycles = 0;
    int64_t totalFLOPs = 0;
    int64_t totalBytes = 0;

    std::map<HWUnit, int64_t> unitCycles;
    std::map<std::string, int64_t> opCounts;
    std::map<std::string, int64_t> opCycles;

    for (int i = 0; i <= static_cast<int>(HWUnit::Scalar); ++i) {
      unitCycles[static_cast<HWUnit>(i)] = 0;
    }

    module.walk([&](Operation *op) {
      auto cyclesAttr = op->getAttrOfType<IntegerAttr>("estimated_cycles");
      if (!cyclesAttr)
        return;

      int64_t cycles = cyclesAttr.getInt();
      std::string opName = op->getName().getStringRef().str();

      opCounts[opName]++;
      opCycles[opName] += cycles;

      HWUnit hwUnit = getOpHWUnit(op);

      // Count FLOPs
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        int64_t m = matmulOp.getM();
        int64_t n = matmulOp.getN();
        int64_t k = matmulOp.getK();
        totalFLOPs += 2 * m * n * k;
      } else if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp, NegOp, AbsOp,
                     ReluOp>(op)) {
        if (op->getNumResults() > 0) {
          totalFLOPs += getNumElementsFromType(op->getResult(0).getType());
        }
      } else if (isa<ExpOp, LogOp, SqrtOp, RsqrtOp, TanhOp, SigmoidOp>(op)) {
        if (op->getNumResults() > 0) {
          totalFLOPs += getNumElementsFromType(op->getResult(0).getType()) * 10;
        }
      } else if (isa<ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp>(op)) {
        if (op->getNumOperands() > 0) {
          totalFLOPs += getNumElementsFromType(op->getOperand(0).getType());
        }
      }

      // Count bytes
      if (isa<CubeLoadOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      } else if (isa<CubeStoreOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      } else if (isa<VectorLoadOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      } else if (isa<VectorStoreOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      }

      unitCycles[hwUnit] += cycles;
      totalCycles += cycles;
    });

    auto scheduledCyclesAttr =
        module->getAttrOfType<IntegerAttr>("ascend.scheduled_cycles");
    auto rooflineCyclesAttr =
        module->getAttrOfType<IntegerAttr>("ascend.roofline_cycles");
    int64_t scheduledCycles =
        scheduledCyclesAttr
            ? scheduledCyclesAttr.getInt()
            : (rooflineCyclesAttr ? rooflineCyclesAttr.getInt() : totalCycles);

    // Ascend 910B clock: 1.85 GHz = 1850 cycles/us
    constexpr double CYCLES_PER_US = 1850.0;
    double timeUs = scheduledCycles / CYCLES_PER_US;
    double achievedTFLOPS = (totalFLOPs / 1e12) / (timeUs / 1e6);
    double achievedBandwidth = (totalBytes / 1e9) / (timeUs / 1e6);
    double arithmeticIntensity =
        totalBytes > 0 ? static_cast<double>(totalFLOPs) / totalBytes : 0;

    double ridgePoint = CUBE_TFLOPS / HBM_BANDWIDTH_GBS;
    bool isComputeBound = arithmeticIntensity > ridgePoint;

    HWUnit bottleneck = HWUnit::Scalar;
    int64_t maxCycles = 0;
    for (const auto &[unit, cycles] : unitCycles) {
      if (cycles > maxCycles) {
        maxCycles = cycles;
        bottleneck = unit;
      }
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
