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
template <typename ExtractOpT>
bool rewriteOne(Operation *toReplace, mlir::IRMapping &mapper,
                mlir::PatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  if (auto ex = llvm::dyn_cast<ExtractOpT>(toReplace)) {
    rewriter.setInsertionPoint(ex);
    auto newEx = rewriter.create<ExtractOpT>(ex.getLoc(), ex->getResultTypes(),
                                             mapper.lookup(ex.getInput()));
    rewriter.replaceOp(ex, newEx->getResults());
    return true;
  } else {
    return false;
  }
}
template <typename ExtractOpT>
bool mapInputTensorOnce(Operation *toReplace,
                        llvm::SmallDenseSet<Value> &mappedValues) {
  if (auto ex = llvm::dyn_cast<ExtractOpT>(toReplace)) {
    if (auto input = ex.getInput(); isa<RankedTensorType>(input.getType())) {
      mappedValues.insert(input);
      return true;
    }
  }
  return false;
}

template <typename... OpTys>
static bool rewriteExtractWithMappedInput(Operation *toReplace,
                                          IRMapping &mapper,
                                          PatternRewriter &rewriter) {
  return (rewriteOne<OpTys>(toReplace, mapper, rewriter) || ...);
}

template <typename... OpTys>
static bool mapInputTensors(Operation *toReplace,
                            llvm::SmallDenseSet<Value> &mappedValues) {
  return (mapInputTensorOnce<OpTys>(toReplace, mappedValues) || ...);
}

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

struct TleArgConversion : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(LLVM::CallOp op,
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
TleArgConversion::matchAndRewrite(LLVM::CallOp op,
                                  PatternRewriter &rewriter) const {
  SmallVector<Value> operands = op.getOperands();
  PatternRewriter::InsertionGuard guard(rewriter);
  bool hasConversion = false;
  IRMapping mapper;
  SmallVector<Operation *> targets;
  SmallVector<ttg::LocalAllocOp> toDeallocOps;
  llvm::SmallDenseSet<Value> mappedValues;
  for (Value operand : operands) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      continue;
    }
    hasConversion |=
        mapInputTensors<tle::ExtractAllocatedPtrOp, tle::ExtractSizesOp,
                        tle::ExtractStridesOp, tle::ExtractOffsetOp,
                        tle::ExtractAlignedPtrOp>(defOp, mappedValues);
    if (isa<tle::ExtractAllocatedPtrOp, tle::ExtractSizesOp,
            tle::ExtractStridesOp, tle::ExtractOffsetOp,
            tle::ExtractAlignedPtrOp>(defOp)) {
      targets.push_back(defOp);
    }
  }
  for (Value tensorVal : mappedValues) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(tensorVal);
    auto memDesc = getPlainMemDesc(cast<RankedTensorType>(tensorVal.getType()));
    auto localAllocOp =
        rewriter.create<ttg::LocalAllocOp>(tensorVal.getLoc(), memDesc);
    auto localStoreOp = rewriter.create<ttg::LocalStoreOp>(
        localAllocOp.getLoc(), tensorVal, localAllocOp);
    mapper.map(tensorVal, localAllocOp.getResult());
    toDeallocOps.push_back(localAllocOp);
    hasConversion = true;
  }
  SmallVector<tle::PackOp> packs;
  for (Value res : op->getResults()) {
    for (OpOperand &use : res.getUses()) {
      if (auto packop = dyn_cast<tle::PackOp>(use.getOwner()))
        if (auto tensorTy =
                dyn_cast<RankedTensorType>(packop.getOutput().getType())) {
          rewriter.setInsertionPoint(packop);
          auto newPackOp = rewriter.create<tle::PackOp>(
              packop.getLoc(), getPlainMemDesc(tensorTy), packop.getInput());
          auto loadOp = rewriter.create<ttg::LocalLoadOp>(
              newPackOp.getLoc(), tensorTy, newPackOp.getOutput());
          rewriter.replaceOp(packop, loadOp.getResult());
          rewriter.setInsertionPointAfter(loadOp);
          hasConversion = true;
        }
    }
  }
  for (ttg::LocalAllocOp toDeallocOp : toDeallocOps) {
    rewriter.create<ttg::LocalDeallocOp>(toDeallocOp.getLoc(), toDeallocOp);
    hasConversion = true;
  }
  if (!hasConversion) {
    return failure();
  }
  for (Operation *toReplace : targets) {
    rewriteExtractWithMappedInput<
        tle::ExtractAllocatedPtrOp, tle::ExtractSizesOp, tle::ExtractStridesOp,
        tle::ExtractOffsetOp, tle::ExtractAlignedPtrOp>(toReplace, mapper,
                                                        rewriter);
  }
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
