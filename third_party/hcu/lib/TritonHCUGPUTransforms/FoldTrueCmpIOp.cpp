#include "TritonHCUGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/hcu/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUFOLDTRUECMPI
#include "TritonHCUGPUTransforms/Passes.h.inc"

struct TritonHCUFoldTrueCmpIOpPass
    : impl::TritonHCUFoldTrueCmpIBase<TritonHCUFoldTrueCmpIOpPass> {

  void runOnOperation() override {
    DenseMap<Value, SetVector<Operation *>> assumptions =
        HCU::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    ModuleOp mod = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    HCU::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<HCU::TritonIntegerRangeAnalysis>(
            assumptions, &getAnalysis<DominanceInfo>());
    HCU::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(&getContext());
    HCU::populateFoldTrueCmpIOpPatterns(patterns, solver.get());
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace mlir
