#include "epu/memory.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_MEMORYPROMOTION
#include "evas/Transform/Linalg/Passes.h.inc"
namespace mlir::triton::ev {

namespace {

LogicalResult promoteOutput(Operation *op, unsigned outputIndex,
                            mlir::ev::MemScope srcScope,
                            mlir::ev::MemScope dstScope,
                            PatternRewriter &rewriter) {
  if (op->getNumOperands() <= outputIndex) {
    return failure();
  }

  Value outputOperand = op->getOperand(outputIndex);
  MemRefType memRefType = dyn_cast<MemRefType>(outputOperand.getType());
  if (!memRefType) {
    return failure();
  }

  mlir::ev::MemScope currentScope = mlir::ev::getMemScope(memRefType);
  if (currentScope != srcScope) {
    return failure();
  }

  MemRefType promotedType = MemRefType::get(
      memRefType.getShape(), memRefType.getElementType(),
      memRefType.getLayout(),
      rewriter.getI64IntegerAttr(static_cast<int64_t>(dstScope)));

  auto promotedAlloc =
      rewriter.create<memref::AllocOp>(op->getLoc(), promotedType);
  Value promotedValue = Value(promotedAlloc.getResult());

  op->setOperand(outputIndex, promotedValue);
  rewriter.setInsertionPointAfter(op);
  rewriter.create<memref::CopyOp>(op->getLoc(), promotedValue, outputOperand);

  return success();
}

LogicalResult promoteInput(Operation *op, unsigned inputIndex,
                           mlir::ev::MemScope srcScope,
                           mlir::ev::MemScope dstScope,
                           PatternRewriter &rewriter) {
  if (op->getNumOperands() <= inputIndex) {
    return failure();
  }

  Value inputOperand = op->getOperand(inputIndex);
  MemRefType memRefType = dyn_cast<MemRefType>(inputOperand.getType());
  if (!memRefType) {
    return failure();
  }

  mlir::ev::MemScope currentScope = mlir::ev::getMemScope(memRefType);
  if (currentScope != srcScope) {
    return failure();
  }

  MemRefType promotedType = MemRefType::get(
      memRefType.getShape(), memRefType.getElementType(),
      memRefType.getLayout(),
      rewriter.getI64IntegerAttr(static_cast<int64_t>(dstScope)));

  auto promotedAlloc =
      rewriter.create<memref::AllocOp>(op->getLoc(), promotedType);
  rewriter.create<memref::CopyOp>(op->getLoc(), inputOperand, promotedAlloc);

  op->setOperand(inputIndex, promotedAlloc);

  return success();
}

struct MatmulMemoryPromotionPattern
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    return promoteOutput(matmulOp.getOperation(), 2, mlir::ev::MemScope::MM,
                   mlir::ev::MemScope::FAM, rewriter);
  }
};

struct PromoteLinalgOperandsToMMPattern : public RewritePattern {
  PromoteLinalgOperandsToMMPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto *dialect = op->getDialect();
    if (!dialect ||
        dialect->getNamespace() != linalg::LinalgDialect::getDialectNamespace())
      return failure();
    // (TODO) Tricky code here, since takeop and scatterop's first input always from ddr
    if (isa<linalg::TakeOp, linalg::ScatterOp>(op)) return failure();
    bool promotedAny = false;
    auto linalgOp = dyn_cast<DestinationStyleOpInterface>(op);
    unsigned numInputs =
        linalgOp ? linalgOp.getNumDpsInputs() : op->getNumOperands();

    for (auto it : llvm::enumerate(op->getOperands())) {
      Value operand = it.value();
      auto memRefType = dyn_cast<MemRefType>(operand.getType());
      if (!memRefType)
        continue;

      mlir::ev::MemScope currentScope = mlir::ev::getMemScope(memRefType);
      if (currentScope >= mlir::ev::MemScope::MM)
        continue;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      LogicalResult result = failure();
      if (it.index() < numInputs) {
        result = promoteInput(op, it.index(), currentScope,
                              mlir::ev::MemScope::MM, rewriter);
      } else {
        result = promoteOutput(op, it.index(), currentScope,
                               mlir::ev::MemScope::MM, rewriter);
      }
      if (succeeded(result)) {
        promotedAny = true;
      }
    }

    return promotedAny ? success() : failure();
  }
};
} // namespace

struct MemoryPromotionPass
    : public ::impl::MemoryPromotionBase<MemoryPromotionPass> {
  using MemoryPromotionBase<MemoryPromotionPass>::MemoryPromotionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulMemoryPromotionPattern>(
        &getContext());
    patterns.add<PromoteLinalgOperandsToMMPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createMemoryPromotionPass() {
  return std::make_unique<MemoryPromotionPass>();
}

} // namespace mlir::triton::ev
