#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/SqmmaOperandPlan.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUOPTIMIZEDOTOPERANDS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

static bool isDescriptorTensorViewChain(Value value) {
  while (value) {
    if (value.getDefiningOp<tt::DescriptorLoadOp>())
      return true;
    if (auto transOp = value.getDefiningOp<tt::TransOp>()) {
      value = transOp.getSrc();
      continue;
    }
    if (auto reshapeOp = value.getDefiningOp<tt::ReshapeOp>()) {
      value = reshapeOp.getSrc();
      continue;
    }
    return false;
  }
  return false;
}

static bool isSwizzledSharedMemDesc(ttg::MemDescType type) {
  return isa_and_nonnull<ttg::SwizzledSharedEncodingAttr>(type.getEncoding());
}

static bool shouldNormalizeViewLocalAlloc(ttg::LocalAllocOp allocOp,
                                          Value viewSource,
                                          ttg::MemDescType sourceTy,
                                          ttg::MemDescType finalViewTy) {
  if (!triton::musa::hasSqmmaOpIdxAttr(allocOp.getOperation()) &&
      !isDescriptorTensorViewChain(viewSource))
    return false;

  if (isSwizzledSharedMemDesc(finalViewTy) &&
      !isSwizzledSharedMemDesc(sourceTy))
    return false;

  return true;
}

static bool isDotLikeUserForSwizzle(Operation *op) {
  return isa<tt::DotOp, tt::musa::WmmaDotOp>(op);
}

static void resetWmmaLayoutForMaterializedTranspose(Value dotOperand,
                                                    Operation *user) {
  auto wmma = dyn_cast<tt::musa::WmmaDotOp>(user);
  if (!wmma)
    return;

  if (wmma.getA() == dotOperand)
    wmma.setLayoutA(triton::musa::SQMMALayout::row);
  if (wmma.getB() == dotOperand)
    wmma.setLayoutB(triton::musa::SQMMALayout::col);
}

static RankedTensorType getSharedLayoutSourceType(tt::TransOp trans) {
  RankedTensorType srcTy = trans.getSrc().getType();
  if (auto srcCvt = trans.getSrc().getDefiningOp<ttg::ConvertLayoutOp>())
    srcTy = srcCvt.getSrc().getType();
  return srcTy;
}

static FailureOr<Value> createSwizzledTransLocalLoad(
    tt::TransOp trans, Value src, RankedTensorType srcTy,
    RankedTensorType sharedLoadTy, PatternRewriter &rewriter) {
  auto cvtEncoding =
      dyn_cast<ttg::DotOperandEncodingAttr>(sharedLoadTy.getEncoding());
  if (!cvtEncoding)
    return failure();

  auto *ctx = rewriter.getContext();
  auto oldCGALayout = ttg::getCGALayout(srcTy.getEncoding());
  auto newLl =
      transposeLinearLayout(oldCGALayout.getLinearLayout(), trans.getOrder());
  auto newCGALayout = ttg::CGAEncodingAttr::get(ctx, std::move(newLl));
  auto newInnerCvtEnc = triton::musa::composeMusaOperandSharedLayout(
      cvtEncoding, srcTy.getShape(),
      /*order=*/ttg::getOrderForMemory(srcTy), newCGALayout,
      srcTy.getElementType(),
      /*needTrans=*/true);
  if (!newInnerCvtEnc)
    return failure();

  rewriter.setInsertionPoint(trans);
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto alloc = ttg::LocalAllocOp::create(
      rewriter, trans.getLoc(),
      ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                            *newInnerCvtEnc, sharedMemorySpace),
      src);
  auto newTrans = ttg::MemDescTransOp::create(rewriter, trans.getLoc(), alloc,
                                              ArrayRef<int32_t>({1, 0}));
  return ttg::LocalLoadOp::create(rewriter, trans.getLoc(), sharedLoadTy,
                                  newTrans)
      .getResult();
}

class SwizzleShmemConvert : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
#ifdef __TLE__
    if (isTleExplicitConvertLayoutOp(cvtOp))
      return failure();
#endif // __TLE__
    if (!cvtOp->hasOneUse() ||
        !isDotLikeUserForSwizzle(cvtOp->use_begin()->getOwner()))
      return failure();
    auto trans = cvtOp.getSrc().getDefiningOp<tt::TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0} ||
        !trans->hasOneUse())
      return failure();

    RankedTensorType sharedLoadTy = cvtOp.getType();
    auto localLoad = createSwizzledTransLocalLoad(
        trans, trans.getSrc(), getSharedLayoutSourceType(trans), sharedLoadTy,
        rewriter);
    if (failed(localLoad))
      return failure();

    resetWmmaLayoutForMaterializedTranspose(cvtOp.getResult(),
                                            cvtOp->use_begin()->getOwner());

    rewriter.modifyOpInPlace(
        cvtOp, [&]() { cvtOp.getSrcMutable().assign(*localLoad); });
    return success();
  }
};

class SwizzleShmemTrans : public OpRewritePattern<tt::TransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::TransOp trans,
                                PatternRewriter &rewriter) const override {
#ifdef __TLE__
    if (getTleExplicitValueEncoding(trans.getResult()))
      return failure();
#endif // __TLE__
    if (!trans->hasOneUse() ||
        !isDotLikeUserForSwizzle(trans->use_begin()->getOwner()))
      return failure();
    if (trans.getOrder() != ArrayRef<int32_t>{1, 0})
      return failure();

    RankedTensorType sharedLoadTy = trans.getType();
    auto localLoad = createSwizzledTransLocalLoad(
        trans, trans.getSrc(), getSharedLayoutSourceType(trans), sharedLoadTy,
        rewriter);
    if (failed(localLoad))
      return failure();

    resetWmmaLayoutForMaterializedTranspose(trans.getResult(),
                                            trans->use_begin()->getOwner());
    rewriter.replaceOp(trans, *localLoad);
    return success();
  }
};

class NormalizeViewLocalAlloc : public OpRewritePattern<ttg::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc())
      return failure();

    auto allocTy = dyn_cast<ttg::MemDescType>(allocOp.getType());
    if (!allocTy)
      return failure();

    SmallVector<triton::musa::SqmmaTensorViewStep> reverseChain;
    Value base = allocOp.getSrc();
    while (Operation *defOp = base.getDefiningOp()) {
      if (auto transOp = dyn_cast<tt::TransOp>(defOp)) {
        reverseChain.push_back(
            {triton::musa::SqmmaTensorViewKind::Trans, defOp});
        base = transOp.getSrc();
        continue;
      }
      if (auto reshapeOp = dyn_cast<tt::ReshapeOp>(defOp)) {
        reverseChain.push_back(
            {triton::musa::SqmmaTensorViewKind::Reshape, defOp});
        base = reshapeOp.getSrc();
        continue;
      }
      if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(defOp)) {
#ifdef __TLE__
        if (isTleExplicitConvertLayoutOp(cvtOp))
          return failure();
#endif // __TLE__
        reverseChain.push_back(
            {triton::musa::SqmmaTensorViewKind::ConvertLayout, defOp});
        base = cvtOp.getSrc();
        continue;
      }
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
        if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
          return failure();
        reverseChain.push_back(
            {triton::musa::SqmmaTensorViewKind::UnrealizedCast, defOp});
        base = castOp->getOperand(0);
        continue;
      }
      break;
    }
    if (reverseChain.empty())
      return failure();

    SmallVector<triton::musa::SqmmaTensorViewStep> chain(reverseChain.rbegin(),
                                                         reverseChain.rend());
    auto sourceTy =
        triton::musa::inferSqmmaOperandSourceMemDescType(chain, allocTy);
    if (failed(sourceTy))
      return failure();
    if (!shouldNormalizeViewLocalAlloc(allocOp, base, *sourceTy, allocTy))
      return failure();

    auto simulatedFinalTy =
        triton::musa::inferSqmmaOperandFinalMemDescType(chain, *sourceTy);
    if (failed(simulatedFinalTy))
      return failure();
    if (*simulatedFinalTy != allocTy &&
        !triton::musa::areMemDescTypesCompatible(*simulatedFinalTy, allocTy) &&
        !triton::musa::areMemDescTypesLayoutEquivalent(*simulatedFinalTy,
                                                       allocTy))
      return failure();

    auto newAlloc =
        ttg::LocalAllocOp::create(rewriter, allocOp.getLoc(), *sourceTy, base);
    SmallVector<Operation *> createdOps{newAlloc.getOperation()};
    triton::musa::copySqmmaAttrs(allocOp.getOperation(),
                                 newAlloc.getOperation());

    Value current = newAlloc;
    for (const triton::musa::SqmmaTensorViewStep &step : chain) {
      if (step.kind == triton::musa::SqmmaTensorViewKind::Trans) {
        current = triton::musa::materializeTransformedMemDesc(
            rewriter, cast<tt::TransOp>(step.op), current,
            allocOp.getOperation());
        createdOps.push_back(current.getDefiningOp());
        continue;
      }
      if (step.kind == triton::musa::SqmmaTensorViewKind::Reshape) {
        current = triton::musa::materializeReshapedMemDesc(
            rewriter, cast<tt::ReshapeOp>(step.op), current,
            allocOp.getOperation());
        createdOps.push_back(current.getDefiningOp());
        continue;
      }
    }

    if (current.getType() != allocTy) {
      Value adapted = triton::musa::adaptMemDescValue(
          rewriter, allocOp.getLoc(), current, allocTy, allocOp.getOperation());
      if (adapted) {
        current = adapted;
      } else {
        for (Operation *op : llvm::reverse(createdOps)) {
          if (op && op->use_empty())
            rewriter.eraseOp(op);
        }
        return failure();
      }
    }
    tt::replaceUsesAndPropagateType(rewriter, allocOp, current);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

struct TritonMUSAGPUOptimizeDotOperandsPass
    : impl::TritonMUSAGPUOptimizeDotOperandsBase<
          TritonMUSAGPUOptimizeDotOperandsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();

    RewritePatternSet patterns(context);
    patterns
        .add<SwizzleShmemConvert, SwizzleShmemTrans, NormalizeViewLocalAlloc>(
            context);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
