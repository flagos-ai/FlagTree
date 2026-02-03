#include "tle/dialect/include/Transforms/ConvertArgToMemDesc.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TLECONVERTARGTOMEMDESC
#include "tle/dialect/include/Transforms/Passes.h.inc"
} // namespace mlir::triton::tle

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

namespace {

ttg::MemDescType getPlainMemDesc(RankedTensorType ty) {
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  llvm::iota_range<uint32_t> rOrderRange =
      llvm::iota_range<uint32_t>(0, ty.getRank(), false);
  llvm::SmallVector<uint32_t> order = ttg::getOrder(ty);
  return ttg::MemDescType::get(ty.getShape(), ty.getElementType(),
                               ttg::SwizzledSharedEncodingAttr::get(
                                   ty.getContext(), 1, 1, 1, order, ctaLayout),
                               ttg::SharedMemorySpaceAttr::get(ty.getContext()),
                               true);
}

struct TleArgConversion : public OpRewritePattern<tle::ExtractAlignedPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::ExtractAlignedPtrOp op,
                                PatternRewriter &rewriter) const override;
};

struct TleConvertArgToMemDesc
    : public tle::impl::TleConvertArgToMemDescBase<TleConvertArgToMemDesc> {
  void runOnOperation() override;
};

} // namespace

TleArgConversion::TleArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TleArgConversion::matchAndRewrite(tle::ExtractAlignedPtrOp op,
                                  PatternRewriter &rewriter) const {
  // 获取输入
  Value input = op.getInput();
  
  // 检查输入是否是 RankedTensorType
  RankedTensorType tensorTy = dyn_cast<RankedTensorType>(input.getType());
  if (!tensorTy) {
    return failure();
  }
  
  // 在 ExtractAlignedPtrOp 之前创建 LocalAllocOp 和 LocalStoreOp
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  
  ttg::LocalAllocOp allocOp = rewriter.create<ttg::LocalAllocOp>(
      op.getLoc(), getPlainMemDesc(tensorTy));
  rewriter.create<ttg::LocalStoreOp>(op.getLoc(), input, allocOp);
  
  // 创建新的 ExtractAlignedPtrOp，输入改为 MemDesc
  tle::ExtractAlignedPtrOp newOp = rewriter.create<tle::ExtractAlignedPtrOp>(
      op.getLoc(), op.getResult().getType(), allocOp);
  
  // 在 ExtractAlignedPtrOp 之后释放内存
  rewriter.setInsertionPointAfter(newOp);
  rewriter.create<ttg::LocalDeallocOp>(op.getLoc(), allocOp);
  
  // 替换原操作
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

void mlir::triton::tle::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TleArgConversion>(patterns.getContext());
}

void TleConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tle::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
