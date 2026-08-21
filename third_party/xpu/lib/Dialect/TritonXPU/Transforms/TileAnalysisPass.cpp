//===----------------------------------------------------------------------===//
// tritonxpu-tile-analysis -- report-only pass over TileAnalysis.
//
// Gives the pressure measurement a pipeline slot of its own, so the transform
// that consumes it can eventually be handed the result instead of recomputing
// it (redesign-v2.md §4.2 step 1.3, contract in 1.6). Byte equivalence of the
// emitted code is its exit gate: the only thing it ever writes is the removal
// of the keys the M6 probe left behind, and the probe is off by default.
//
// What it can and cannot see: the geometry and the block-wide pressure are
// readable from the IR alone, but the *per-tree* pressure UnrollControl also
// uses comes from the op tree that pass builds while walking (getDAG /
// getOpChainBwd), which is not reproducible here. Reporting the block half is
// still the point of interest -- `decideIterNum` takes
// max(treeP.vecPeak, blockP.vecPeak), so the gap between the two numbers says
// which half dominates at each site. Measured, it goes both ways: on
// layernorm's reduce-for site the tree is 48 against a block of 24, while the
// pointwise sites match exactly. So the block figure is not an upper bound and
// must not be treated as one.
//
// One block-granularity trap the report makes visible: block-wide minVecWidth
// can be 1 (softmax) where the per-tree value is unconstrained, and since
// isLegalIterNum requires minVecWidth % iterNum == 0, that collapses the legal
// set to {1}. Widening the *target* to block scope is not the same as widening
// minVecWidth to block scope.
//===----------------------------------------------------------------------===//

#include "triton/Analysis/TileAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "triton/Tools/Sys/GetEnv.hpp"

#define DEBUG_TYPE "tritonxpu-tile-analysis"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUTILEANALYSIS
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// Same legality test as UnrollControl::isLegalIterNum. Duplicated rather than
// shared because sharing it means moving it into the analysis unit, which is a
// logic move and belongs to a later step -- this pass is only allowed to
// observe. When the predicate does move, this copy goes away.
bool isLegalIterNum(int64_t iterNum, int64_t numCol, int64_t widthPerCore,
                    int64_t minVecWidth) {
  return iterNum >= 1 && numCol % iterNum == 0 && widthPerCore % iterNum == 0 &&
         (minVecWidth == 0 || minVecWidth % iterNum == 0);
}

} // namespace

struct TritonXPUTileAnalysisPass
    : public impl::TritonXPUTileAnalysisBase<TritonXPUTileAnalysisPass> {

public:
  using impl::TritonXPUTileAnalysisBase<
      TritonXPUTileAnalysisPass>::TritonXPUTileAnalysisBase;

  TritonXPUTileAnalysisPass() = default;
  TritonXPUTileAnalysisPass(unsigned vrfBudget) { this->vrfBudget = vrfBudget; }

  void report(const char *site, Operation *op, Type valTy) {
    auto tensorTy = dyn_cast<RankedTensorType>(valTy);
    if (!tensorTy)
      return;
    int64_t numCol = tensorTy.getShape().back();
    int64_t widthPerCore = numCol, coresPerGroup = 1;
    if (auto clusterEncoding = getClusterLayout(tensorTy)) {
      coresPerGroup = clusterEncoding.getCoresPerGroup().back();
      widthPerCore = clusterEncoding.getSizePerCore().back();
    }

    RegPressure p;
    getBlockRegPressure(getOperation(), op, p);

    int64_t target =
        this->vrfBudget > 0 ? ceil<int64_t>(p.vecPeak, this->vrfBudget) : 1;
    target = std::max<int64_t>(target, 1);

    // The legal set, printed in full: its sparseness is the reason step 3.6
    // exists, and a report that only gave the chosen factor would hide it.
    std::string legal;
    for (int64_t iterNum = 1; iterNum <= widthPerCore; ++iterNum)
      if (isLegalIterNum(iterNum, numCol, widthPerCore, p.minVecWidth))
        legal += (legal.empty() ? "" : ",") + std::to_string(iterNum);

    StringRef kernel = "<unknown>";
    if (auto funcOp = op->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();

    llvm::errs() << "[TileAnalysis] " << kernel << " site=" << site
                 << " numCol=" << numCol << " widthPerCore=" << widthPerCore
                 << " coresPerGroup=" << coresPerGroup
                 << " blockVecPeak=" << p.vecPeak
                 << " blockVecTotal=" << p.vecTotal
                 << " blockScalarPeak=" << p.scalarPeak
                 << " maxVecWidth=" << p.maxVecWidth
                 << " minVecWidth=" << p.minVecWidth
                 << " budget=" << this->vrfBudget << " blockTarget=" << target
                 << " legal={" << legal << "}\n";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // The consumer end of the M6 contract (step 1.6). This pass sits
    // immediately before `tritonxpu-unroll-control`, which is where M4/M5 will
    // read the plan, so it is the right place to ask whether either key still
    // finds its root. Runs before the report gate and on its own switch,
    // because it also erases what the probe wrote. (It is registered under
    // `isCloseUnrollControl`, so with unroll control off the probe simply
    // produces no report -- the writes are gated on the same env var, so
    // nothing is left behind either way.)
    tilePlanCheck(mod);

    // Cheap by default: the walk below is pure measurement, so it is only worth
    // paying for when someone is reading the report.
    if (!mlir::triton::tools::getBoolEnv("TRITONXPU_TILE_REPORT"))
      return;

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      report("pointwise", storeOp, storeOp.getValue().getType());
    });
    mod.walk([&](triton::xpu::ReduceOp reduceOp) {
      report("reduce", reduceOp, reduceOp.getInputTypes()[0]);
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
