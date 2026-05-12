#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
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

static SmallVector<int64_t>
invertTrailingPermutation(ArrayRef<int64_t> allocShape,
                          ArrayRef<int32_t> order) {
  SmallVector<int64_t> result;
  auto rank = static_cast<size_t>(order.size());
  if (allocShape.size() < rank)
    return result;
  result.assign(allocShape.begin(), allocShape.end() - rank);

  SmallVector<int64_t> tail(allocShape.end() - rank, allocShape.end());
  SmallVector<int64_t> inverse(rank);
  for (auto [idx, permutedIdx] : llvm::enumerate(order))
    inverse[permutedIdx] = tail[idx];
  result.append(inverse.begin(), inverse.end());
  return result;
}

class SwizzleShmemConvert : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    if (!cvtOp->hasOneUse() || !isa<tt::DotOp>(cvtOp->use_begin()->getOwner()))
      return failure();
    auto trans = cvtOp.getSrc().getDefiningOp<tt::TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0} ||
        !trans->hasOneUse())
      return failure();

    RankedTensorType srcTy = trans.getSrc().getType();
    if (auto srcCvt = trans.getSrc().getDefiningOp<ttg::ConvertLayoutOp>())
      srcTy = srcCvt.getSrc().getType();

    RankedTensorType sharedLoadTy = cvtOp.getType();
    auto cvtEncoding =
        dyn_cast<ttg::DotOperandEncodingAttr>(sharedLoadTy.getEncoding());
    if (!cvtEncoding)
      return failure();

    auto *ctx = getContext();
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
        trans.getSrc());
    auto newTrans = ttg::MemDescTransOp::create(rewriter, trans.getLoc(), alloc,
                                                ArrayRef<int32_t>({1, 0}));
    auto localLoadOp = ttg::LocalLoadOp::create(rewriter, trans.getLoc(),
                                                sharedLoadTy, newTrans);
    rewriter.modifyOpInPlace(cvtOp, [&]() {
      cvtOp.getSrcMutable().assign(localLoadOp.getResult());
    });
    return success();
  }
};

class NormalizeDescriptorTransLocalAlloc
    : public OpRewritePattern<ttg::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc())
      return failure();
    auto transOp = allocOp.getSrc().getDefiningOp<tt::TransOp>();
    if (!transOp || !isDescriptorTensorViewChain(transOp.getSrc()))
      return failure();

    auto allocTy = dyn_cast<ttg::MemDescType>(allocOp.getType());
    auto srcTy = dyn_cast<RankedTensorType>(transOp.getSrc().getType());
    if (!allocTy || !srcTy || !allocTy.getEncoding())
      return failure();

    Dialect &dialect = allocTy.getEncoding().getDialect();
    auto inferLayoutInterface =
        dyn_cast<tt::DialectInferLayoutInterface>(&dialect);
    if (!inferLayoutInterface)
      return failure();

    Attribute sourceEncoding;
    if (failed(inferLayoutInterface->inferTransOpEncoding(
            allocTy.getEncoding(), srcTy.getShape(), transOp.getOrder(),
            sourceEncoding, allocOp.getLoc()))) {
      return failure();
    }

    SmallVector<int64_t> sourceAllocShape =
        invertTrailingPermutation(allocTy.getAllocShape(), transOp.getOrder());
    if (sourceAllocShape.empty())
      return failure();

    auto sourceTy = ttg::MemDescType::get(
        srcTy.getShape(), srcTy.getElementType(), sourceEncoding,
        allocTy.getMemorySpace(), allocTy.getMutableMemory(), sourceAllocShape);

    auto newAlloc = ttg::LocalAllocOp::create(rewriter, allocOp.getLoc(),
                                              sourceTy, transOp.getSrc());
    triton::musa::copySqmmaAttrs(allocOp.getOperation(),
                                 newAlloc.getOperation());

    Value transposed = ttg::MemDescTransOp::create(
        rewriter, transOp.getLoc(), newAlloc, transOp.getOrder());
    triton::musa::copySqmmaAttrs(allocOp.getOperation(),
                                 transposed.getDefiningOp());
    rewriter.replaceOp(allocOp, transposed);
    return success();
  }
};

class NormalizeDescriptorReshapeLocalAlloc
    : public OpRewritePattern<ttg::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc())
      return failure();
    auto reshapeOp = allocOp.getSrc().getDefiningOp<tt::ReshapeOp>();
    if (!reshapeOp || !isDescriptorTensorViewChain(reshapeOp.getSrc()))
      return failure();

    auto allocTy = dyn_cast<ttg::MemDescType>(allocOp.getType());
    auto srcTy = dyn_cast<RankedTensorType>(reshapeOp.getSrc().getType());
    if (!allocTy || !srcTy)
      return failure();

    ttg::MemDescType sourceTy;
    if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
            getContext(), allocOp.getLoc(), allocTy, srcTy.getShape(),
            sourceTy))) {
      return failure();
    }

    auto newAlloc = ttg::LocalAllocOp::create(rewriter, allocOp.getLoc(),
                                              sourceTy, reshapeOp.getSrc());
    triton::musa::copySqmmaAttrs(allocOp.getOperation(),
                                 newAlloc.getOperation());

    Value reshaped = ttg::MemDescReshapeOp::create(
        rewriter, reshapeOp.getLoc(), newAlloc, allocTy.getShape());
    if (reshaped.getType() != allocOp.getType()) {
      reshaped.getDefiningOp()->erase();
      newAlloc.erase();
      return failure();
    }
    triton::musa::copySqmmaAttrs(allocOp.getOperation(),
                                 reshaped.getDefiningOp());
    rewriter.replaceOp(allocOp, reshaped);
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
    patterns.add<SwizzleShmemConvert, NormalizeDescriptorTransLocalAlloc,
                 NormalizeDescriptorReshapeLocalAlloc>(context);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
