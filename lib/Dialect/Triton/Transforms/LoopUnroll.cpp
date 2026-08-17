#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    SmallVector<scf::ForOp, 4> noUnrollLoops;
    getOperation()->walk([&](scf::ForOp forOp) {
      auto factor =
          forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName);
      if (!factor)
        return;
      if (factor.getInt() > 1)
        loops.push_back(forOp);
      else
        noUnrollLoops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    Builder builder(ctx);
    auto disabledUnroll = LLVM::LoopUnrollAttr::get(
        ctx, builder.getBoolAttr(true), {}, {}, {}, {}, {}, {});
    auto noUnrollAnnotation = LLVM::LoopAnnotationAttr::get(
        ctx, {}, {}, {}, disabledUnroll, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {});
    for (auto loop : noUnrollLoops) {
      loop->removeAttr(loopUnrollFactorAttrName);
      loop->setAttr("llvm.loop_annotation", noUnrollAnnotation);
    }

    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);
      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // namespace mlir::triton
