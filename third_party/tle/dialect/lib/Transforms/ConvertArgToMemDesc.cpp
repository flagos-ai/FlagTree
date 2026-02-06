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
void rewriteExtractWithMappedInput(
    Operation *toReplace,
    mlir::IRMapping &mapper,
    mlir::PatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  if(auto ex = llvm::dyn_cast<ExtractOpT>(toReplace)){
    rewriter.setInsertionPoint(ex);
    auto newEx = rewriter.create<ExtractOpT>(
      ex.getLoc(), ex->getResultTypes(), mapper.lookup(ex.getInput()));
    rewriter.replaceOp(ex, newEx->getResults());
  }
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

struct TleArgConversion : public OpRewritePattern<tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::DSLRegionOp  op,
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
TleArgConversion::matchAndRewrite(tle::DSLRegionOp op,
                                  PatternRewriter &rewriter) const {
  SmallVector<Value> inputs(op.getInputs().begin(),
                                 op.getInputs().end());
  SmallVector<Value> outputs(op.getOutputs().begin(),
                                  op.getOutputs().end());
  PatternRewriter::InsertionGuard guard(rewriter);
  SmallVector<Value> operands=llvm::to_vector(llvm::concat<Value>(outputs, inputs));
  bool hasConversion = false;
  IRMapping mapper;
  SmallVector<Operation *> targets;
  SmallVector<ttg::LocalAllocOp> toDeallocOps;
  for(Value dslValue : operands) {
    Operation *defOp = dslValue.getDefiningOp();
    if (!defOp) continue;
    if (auto ex = dyn_cast<tle::ExtractAllocatedPtrOp>(defOp)) {
      if(auto tensorTy = dyn_cast<RankedTensorType>(ex.getInput().getType())){
        rewriter.setInsertionPoint(ex);
        auto allocOp = rewriter.create<ttg::LocalAllocOp>(ex.getLoc(),
                                                    getPlainMemDesc(tensorTy));
        rewriter.create<ttg::LocalStoreOp>(
          ex.getLoc(), ex.getInput(), allocOp);
        auto newAligned = rewriter.create<tle::ExtractAllocatedPtrOp>(
        ex.getLoc(), ex.getResult().getType(), allocOp);
        mapper.map(ex.getInput(), allocOp.getResult());
        rewriter.replaceOp(ex, newAligned.getResult());
        rewriter.setInsertionPointAfter(op);
        toDeallocOps.push_back(allocOp);
        hasConversion = true;
        }
      }
    if(isa<tle::ExtractSizesOp, tle::ExtractStridesOp, tle::ExtractOffsetOp,tle::ExtractAlignedPtrOp>(defOp)){
        targets.push_back(defOp);
      }
    }
  SmallVector<tle::PackOp> packs;
  for (Value res : op->getResults()) {
    for (OpOperand &use : res.getUses()) {
      if (auto pack = dyn_cast<tle::PackOp>(use.getOwner()))
        packs.push_back(pack);
    }
  }
  for(tle::PackOp packop: packs) {
    if(auto tensorTy = dyn_cast<RankedTensorType>(packop.getOutput().getType())){
      rewriter.setInsertionPoint(packop);
      auto newPackOp = rewriter.create<tle::PackOp>(
        packop.getLoc(), getPlainMemDesc(tensorTy), packop.getInput());
      auto loadOp = rewriter.create<ttg::LocalLoadOp>(newPackOp.getLoc(), tensorTy,
                                                newPackOp.getOutput());
      rewriter.replaceOp(packop, loadOp.getResult());
      rewriter.setInsertionPointAfter(loadOp);
      hasConversion = true;
      }
  }
  for(ttg::LocalAllocOp toDeallocOp : toDeallocOps) {
    rewriter.create<ttg::LocalDeallocOp>(toDeallocOp.getLoc(), toDeallocOp);
  }
  if(!hasConversion) {
      return failure();
  }
  for(Operation *toReplace : targets) {
      rewriteExtractWithMappedInput<tle::ExtractSizesOp>(toReplace, mapper, rewriter);
      rewriteExtractWithMappedInput<tle::ExtractStridesOp>(toReplace, mapper, rewriter);
      rewriteExtractWithMappedInput<tle::ExtractOffsetOp>(toReplace, mapper, rewriter);
      rewriteExtractWithMappedInput<tle::ExtractAlignedPtrOp>(toReplace, mapper, rewriter);
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
