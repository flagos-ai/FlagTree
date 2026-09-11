//===----------------------------------------------------------------------===//
// MKLoopBoundCanonicalizePass
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/MKPipeline/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MKLOOPBOUNDCANONICALIZEPASS
#include "magic-kernel/Conversion/MKPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

std::optional<int64_t> getConstantInt(Value value) {
  APInt intValue;
  if (!matchPattern(value, m_ConstantInt(&intValue)))
    return std::nullopt;
  return intValue.getSExtValue();
}

Value getLoopInductionValue(Value value) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return cast.getIn();
  return value;
}

bool sameValue(Value lhs, Value rhs) {
  return getLoopInductionValue(lhs) == getLoopInductionValue(rhs);
}

Value createIntegerConstant(PatternRewriter &rewriter, Location loc, Type type,
                            int64_t value) {
  if (type.isIndex())
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  return rewriter.create<arith::ConstantOp>(
      loc, type, rewriter.getIntegerAttr(type, value));
}

bool matchAddOfIvAndStep(arith::AddIOp addOp, Value ivLike,
                         int64_t &stepValue) {
  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();

  if (sameValue(lhs, ivLike)) {
    if (auto cst = getConstantInt(rhs)) {
      stepValue = *cst;
      return true;
    }
  }

  if (sameValue(rhs, ivLike)) {
    if (auto cst = getConstantInt(lhs)) {
      stepValue = *cst;
      return true;
    }
  }

  return false;
}

bool matchMinOfAddAndBound(arith::MinSIOp minOp, arith::AddIOp &addOp,
                           int64_t &boundValue) {
  if ((addOp = minOp.getLhs().getDefiningOp<arith::AddIOp>())) {
    if (auto cst = getConstantInt(minOp.getRhs())) {
      boundValue = *cst;
      return true;
    }
  }

  if ((addOp = minOp.getRhs().getDefiningOp<arith::AddIOp>())) {
    if (auto cst = getConstantInt(minOp.getLhs())) {
      boundValue = *cst;
      return true;
    }
  }

  return false;
}

bool matchMaxOfMinAndIv(arith::MaxSIOp maxOp, Value ivLike,
                        arith::MinSIOp &minOp) {
  if (sameValue(maxOp.getLhs(), ivLike)) {
    minOp = maxOp.getRhs().getDefiningOp<arith::MinSIOp>();
    return static_cast<bool>(minOp);
  }

  if (sameValue(maxOp.getRhs(), ivLike)) {
    minOp = maxOp.getLhs().getDefiningOp<arith::MinSIOp>();
    return static_cast<bool>(minOp);
  }

  return false;
}

bool loopHasOnlyFullTiles(scf::ForOp forOp, int64_t clampUpper,
                          int64_t tileStep) {
  auto lower = getConstantInt(forOp.getLowerBound());
  auto upper = getConstantInt(forOp.getUpperBound());
  auto loopStep = getConstantInt(forOp.getStep());
  if (!lower || !upper || !loopStep)
    return false;

  if (*loopStep <= 0 || tileStep <= 0)
    return false;

  // Keep the first pattern intentionally conservative: the tile size must be
  // the loop step and the clamp bound must be the loop upper bound.
  if (tileStep != *loopStep || clampUpper != *upper)
    return false;

  if (*lower >= *upper)
    return true;

  return ((*upper - *lower) % *loopStep) == 0;
}

bool isTailCmp(arith::CmpIOp cmpOp, Value size, int64_t tileStep) {
  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  Value lhs = cmpOp.getLhs();
  Value rhs = cmpOp.getRhs();
  auto matchConstStep = [&](Value value) {
    auto cst = getConstantInt(value);
    return cst && *cst == tileStep;
  };

  return (lhs == size && matchConstStep(rhs)) ||
         (rhs == size && matchConstStep(lhs));
}

// pattern-1
// %end = arith.addi %iv, %step
// %clamped = arith.minsi %end, %ub
// %safe = arith.maxsi %clamped, %iv
// %size = arith.subi %safe, %iv
// %is_tail = arith.cmpi slt, %size, %step
// scf.if %is_tail { ... zero fill ... }
//
// 当 scf.for 满足常量 iv in [lb, ub) step step，并能证明 iv + step <= ub
// 对所有迭代成立，就把 %size 替换成 %step，%is_tail 替换成 false，交给后续
// canonicalize/DCE 删除 scf.if。
struct FullTileTailSizePattern : OpRewritePattern<arith::SubIOp> {
  using OpRewritePattern<arith::SubIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubIOp subOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = subOp->getParentOfType<scf::ForOp>();
    if (!forOp)
      return failure();

    Value ivLike = subOp.getRhs();
    if (getLoopInductionValue(ivLike) != forOp.getInductionVar())
      return failure();

    auto maxOp = subOp.getLhs().getDefiningOp<arith::MaxSIOp>();
    if (!maxOp)
      return failure();

    arith::MinSIOp minOp;
    if (!matchMaxOfMinAndIv(maxOp, ivLike, minOp))
      return failure();

    arith::AddIOp addOp;
    int64_t clampUpper = 0;
    if (!matchMinOfAddAndBound(minOp, addOp, clampUpper))
      return failure();

    int64_t tileStep = 0;
    if (!matchAddOfIvAndStep(addOp, ivLike, tileStep))
      return failure();

    if (!loopHasOnlyFullTiles(forOp, clampUpper, tileStep))
      return failure();

    SmallVector<arith::CmpIOp> tailCmps;
    for (Operation *user : llvm::make_early_inc_range(subOp->getUsers())) {
      if (auto cmpOp = dyn_cast<arith::CmpIOp>(user);
          cmpOp && isTailCmp(cmpOp, subOp.getResult(), tileStep))
        tailCmps.push_back(cmpOp);
    }

    for (arith::CmpIOp cmpOp : tailCmps) {
      rewriter.setInsertionPoint(cmpOp);
      Value falseValue = rewriter.create<arith::ConstantOp>(
          cmpOp.getLoc(), rewriter.getBoolAttr(false));
      rewriter.replaceOp(cmpOp, falseValue);
    }

    rewriter.setInsertionPoint(subOp);
    Value fullTileSize = createIntegerConstant(rewriter, subOp.getLoc(),
                                               subOp.getType(), tileStep);
    rewriter.replaceOp(subOp, fullTileSize);
    return success();
  }
};

void populateMKLoopBoundCanonicalizePatterns(RewritePatternSet &patterns) {
  patterns.add<FullTileTailSizePattern>(patterns.getContext());
}

struct MKLoopBoundCanonicalizePass
    : public triton::impl::MKLoopBoundCanonicalizePassBase<
          MKLoopBoundCanonicalizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMKLoopBoundCanonicalizePatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
