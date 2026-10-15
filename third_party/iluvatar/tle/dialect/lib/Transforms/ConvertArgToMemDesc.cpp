#ifdef __ILUVATAR_TLE__

#include "Transforms/ConvertArgToMemDesc.h"
#include "IR/Dialect.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::iluvatar_tle {
#define GEN_PASS_DEF_TRITONILUVATARTLECONVERTARGTOMEMDESC
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::iluvatar_tle

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace iluvatar_tle = mlir::triton::iluvatar_tle;

namespace {

ttg::MemDescType getPlainMemDesc(RankedTensorType ty) {
  ttg::CTAEncodingAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  llvm::iota_range<uint32_t> rOrderRange =
      llvm::iota_range<uint32_t>(0, ty.getRank(), false);
  llvm::SmallVector<uint32_t> order = ttg::getOrder(ty);
  return ttg::MemDescType::get(ty.getShape(), ty.getElementType(),
                               ttg::SwizzledSharedEncodingAttr::get(
                                   ty.getContext(), 1, 1, 1, order, ctaLayout),
                               ttg::SharedMemorySpaceAttr::get(ty.getContext()),
                               true);
}

struct TleArgConversion : public OpRewritePattern<iluvatar_tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(iluvatar_tle::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct TritonIluvatarTleConvertArgToMemDesc
    : public iluvatar_tle::impl::TritonIluvatarTleConvertArgToMemDescBase<
          TritonIluvatarTleConvertArgToMemDesc> {
  void runOnOperation() override;
};

} // namespace

TleArgConversion::TleArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TleArgConversion::matchAndRewrite(iluvatar_tle::DSLRegionOp op,
                                  PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  IRMapping mapper;
  bool hasConversion = false;
  bool needSync = false;
  for (const auto &operand : op->getOperands()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(operand.getType())) {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      ttg::LocalAllocOp allocOp = rewriter.create<ttg::LocalAllocOp>(
          op->getLoc(), getPlainMemDesc(tensorTy));
      rewriter.create<ttg::LocalStoreOp>(op->getLoc(), operand, allocOp);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<ttg::LocalDeallocOp>(op->getLoc(), allocOp);

      newOperands.push_back(allocOp);
      mapper.map(operand, allocOp);
      hasConversion = true;
      needSync = true;
    } else {
      if (isa<ttg::MemDescType>(operand.getType())) {
        needSync = true;
      }
      newOperands.push_back(operand);
    }
  }
  SmallVector<Type> newRetTys;
  for (auto result : op.getResults()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(result.getType())) {
      newRetTys.push_back(getPlainMemDesc(tensorTy));
      hasConversion = true;
    } else {
      newRetTys.push_back(result.getType());
    }
  }
  if (!hasConversion) {
    return failure();
  }
  if (needSync) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.create<NVVM::Barrier0Op>(op.getLoc());
  }
  iluvatar_tle::DSLRegionOp newOp = rewriter.create<iluvatar_tle::DSLRegionOp>(
      op.getLoc(), newRetTys, newOperands, op.getRegionDialectAttr(),
      op.getArgDialectAttr(), op.getOutputOperandIndicesAttr(),
      op->getAttrOfType<StringAttr>("hint"));
  newOp->setAttrs(op->getAttrs());
  PatternRewriter::InsertionGuard guard(rewriter);
  for (auto [idx, oldBlock] : llvm::enumerate(op.getBody().getBlocks())) {
    Block *newBlock = nullptr;
    if (idx == 0) {
      newBlock = rewriter.createBlock(
          &newOp.getBody(), {}, newOp->getOperandTypes(),
          SmallVector<Location>(newOp->getNumOperands(), op.getLoc()));
    } else {
      newBlock = rewriter.createBlock(
          &newOp.getBody(), {}, oldBlock.getArgumentTypes(),
          SmallVector<Location>(oldBlock.getNumArguments(), op.getLoc()));
    }
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
      mapper.map(oldArg, newArg);
    }
    mapper.map(&oldBlock, newBlock);
  }
  for (auto [oldBlock, newBlock] :
       llvm::zip(op.getBody().getBlocks(), newOp.getBody().getBlocks())) {
    rewriter.setInsertionPointToEnd(&newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      if (iluvatar_tle::PackOp packOp =
              dyn_cast<iluvatar_tle::PackOp>(operation)) {
        if (auto tensorTy =
                dyn_cast<RankedTensorType>(packOp.getOutput().getType())) {
          iluvatar_tle::PackOp newPackOp =
              rewriter.create<iluvatar_tle::PackOp>(
                  packOp.getLoc(), getPlainMemDesc(tensorTy),
                  mapper.lookup(packOp.getInput()));
          mapper.map(packOp.getOutput(), newPackOp.getOutput());
          continue;
        }
      }
      rewriter.clone(operation, mapper);
    }
  }
  rewriter.setInsertionPointAfter(newOp);
  SmallVector<Value> results;
  for (auto [oldResult, newResult] :
       llvm::zip(op.getResults(), newOp.getResults())) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(oldResult.getType())) {
      ttg::LocalLoadOp loadOp =
          rewriter.create<ttg::LocalLoadOp>(op.getLoc(), tensorTy, newResult);
      results.push_back(loadOp);
    } else {
      results.push_back(newResult);
    }
  }
  rewriter.replaceOp(op, results);
  return success();
}

void mlir::triton::iluvatar_tle::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TleArgConversion>(patterns.getContext());
}

void TritonIluvatarTleConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  iluvatar_tle::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

#endif // __ILUVATAR_TLE__
