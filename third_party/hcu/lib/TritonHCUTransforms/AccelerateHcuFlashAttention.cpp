#include "TritonHCUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/hcu/include/TritonHCUGPUToLLVM/TargetUtils.h"
#include "third_party/hcu/include/TritonHCUTransforms/MfmaGroup.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using mlir::arith::DivFOp;
using mlir::arith::TruncFOp;
using triton::BroadcastOp;
using triton::ExpandDimsOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::HCUMfmaEncodingAttr;

void collectDependentDotOps(Operation *op,
                            std::vector<Operation *> &dependentDotOps) {
  for (Value operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      int count =
          std::count(dependentDotOps.begin(), dependentDotOps.end(), defOp);
      if (defOp->getBlock() == op->getBlock() && count <= 0) {
        collectDependentDotOps(defOp, dependentDotOps);
        if (isa<triton::DotOp>(defOp)) {
          dependentDotOps.push_back(defOp);
        }
      }
    }
  }
}

/*
 * 1. transpose=false, mfma->dotOp<0, mfma>
 * mfma: transpose=false, interleave=false
 * mfma1: transpose=true, interleave=true
 *
 * convert[mfma->dotOp<0, mfma>] -> dot<mfma>  ====>>>>  convert[mfma1->dotOp<0,
 * mfma1>] -> dot<mfma1> -> convert[mfma1->mfma] / / convert[xxxx->dotOp<1,
 * mfma>]                         convert[xxxx->dotOp<1, mfma1>]
 *
 * 2. transpose=true, mfma->dotOp<1, mfma>, todo
 * mfma: transpose=true, interleave=false
 * mfma1: transpose=true, interleave=true
 *
 * convert[xxxx->dotOp<0, mfma>] -> dot<mfma>  ====>>>>  convert[xxxx->dotOp<0,
 * mfma1>] -> dot<mfma1> -> convert[mfma1->mfma] / / convert[mfma->dotOp<1,
 * mfma>]                         convert[mfma->dotOp<1, mfma1>]
 */
class TritonHcuFlashAttention : public mlir::RewritePattern {
public:
  TritonHcuFlashAttention(mlir::MLIRContext *context)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 1, context) {}
  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto dotOp = cast<tt::DotOp>(op);

    auto oldD = dotOp.getResult();
    auto oldRetType = cast<RankedTensorType>(oldD.getType());
    auto oldEnc = dyn_cast<HCUMfmaEncodingAttr>(oldRetType.getEncoding());
    if (!oldEnc || oldEnc.getInterleave())
      return mlir::failure();

    auto oldA = dotOp.getA();
    auto oldB = dotOp.getB();
    auto oldAcc = dotOp.getC();

    auto oldAType = cast<RankedTensorType>(oldA.getType());
    auto oldBType = cast<RankedTensorType>(oldB.getType());
    if (!oldAType || !oldBType)
      return mlir::failure();

    if (!oldAType.getElementType().isF16() &&
        !oldAType.getElementType().isBF16())
      return mlir::failure();

    auto cvtAOp = oldA.getDefiningOp<ConvertLayoutOp>();
    if (!cvtAOp)
      return mlir::failure();
    auto cvtAEnc = dyn_cast<HCUMfmaEncodingAttr>(
        cast<RankedTensorType>(cvtAOp.getSrc().getType()).getEncoding());
    if (!cvtAEnc || cvtAEnc.getIsTransposed())
      return mlir::failure(); // A not trans

    auto cvtBOp =
        oldB.getDefiningOp<ConvertLayoutOp>(); // convert xxxx -> dot<1, mfma>
    if (!cvtBOp)
      return mlir::failure();
    auto cvtBEnc = dyn_cast<BlockedEncodingAttr>(
        cast<RankedTensorType>(cvtBOp.getSrc().getType()).getEncoding());
    if (!cvtBEnc)
      return mlir::failure(); // A not trans
    auto bGobalOrder = cvtBEnc.getOrder();

    // auto old_nsB = oldB.getSrc().getDefiningOp<ConvertLayoutOp>(); // convert
    // blocked -> shared if (!old_nsB) return mlir::failure(); auto
    // sharedEncoding = ttg::SharedEncodingAttr::get();

    std::vector<Operation *> dependentDotOps;
    collectDependentDotOps(op, dependentDotOps);
    if (dependentDotOps.empty()) {
      return failure();
    }

    bool isTransposed = false;
    bool interleave = true; // isSecondDot(dotOp);
    unsigned mDim = 16, kDim = 16;
    unsigned nDim = oldBType.getShape()[1] < 32 ? 16 : 32;
    if (bGobalOrder[0] == 0) {
      mDim = 16, nDim = 16;
    }
    auto newEnc = ttg::HCUMfmaEncodingAttr::get(
        ctx, oldEnc.getVersionMajor(), oldEnc.getVersionMinor(),
        oldEnc.getWarpsPerCTA(), mDim, nDim, kDim, isTransposed, interleave,
        oldEnc.getCTALayout());

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(
            ctx, 0, newEnc,
            cast<ttg::DotOperandEncodingAttr>(oldAType.getEncoding())
                .getKWidth()));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(
            ctx, 1, newEnc,
            cast<ttg::DotOperandEncodingAttr>(oldBType.getEncoding())
                .getKWidth()));

    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), newEnc);
    auto newA = rewriter.create<ttg::ConvertLayoutOp>(oldA.getLoc(), newAType,
                                                      cvtAOp.getSrc());
    auto newB = rewriter.create<ttg::ConvertLayoutOp>(oldB.getLoc(), newBType,
                                                      cvtBOp.getSrc());

    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newRetType, newA, newB, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return mlir::success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h.inc"

class TritonHCUAccelerateFlashAttentionPass
    : public TritonHCUAccelerateFlashAttentionBase<
          TritonHCUAccelerateFlashAttentionPass> {
public:
  TritonHCUAccelerateFlashAttentionPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // mlir::PassManager pm(m.getContext());
    // pm.addPass(mlir::createCanonicalizerPass());
    // auto ret = pm.run(m);
    mlir::RewritePatternSet patterns(context);
    patterns.add<TritonHcuFlashAttention>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonHCUAccelerateFlashAttentionPass() {
  return std::make_unique<TritonHCUAccelerateFlashAttentionPass>();
}
