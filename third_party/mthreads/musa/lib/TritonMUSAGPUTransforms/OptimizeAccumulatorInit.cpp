#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

static bool isMusaDotOp(Operation *op) {
  return isa<triton::DotOpInterface>(op) &&
         isa<triton::musa::SquadDotOp, triton::musa::WmmaDotOp>(op);
}

static bool supportsAccumulatorInitOptimization(Operation *op) {
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
    return !sqmma.needsPartialAccumulator();
  }
  if (auto wmma = dyn_cast<triton::musa::WmmaDotOp>(op))
    return !wmma.needsPartialAccumulator();
  return false;
}

static Value getAccumulatorValue(Operation *op) {
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op))
    return sqmma.getC();
  if (auto wmma = dyn_cast<triton::musa::WmmaDotOp>(op))
    return wmma.getC();
  return {};
}

static Value getUseCValue(Operation *op) {
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op))
    return sqmma.getUseC();
  if (auto wmma = dyn_cast<triton::musa::WmmaDotOp>(op))
    return wmma.getUseC();
  return {};
}

static void setUseCValue(Operation *op, Value useC) {
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
    sqmma.getUseCMutable().assign(useC);
    return;
  }
  if (auto wmma = dyn_cast<triton::musa::WmmaDotOp>(op)) {
    wmma.getUseCMutable().assign(useC);
    return;
  }
  llvm_unreachable("unexpected MUSA dot op");
}

static bool isConstantZeroTensor(Value v) {
  return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat());
}

static std::optional<std::pair<Operation *, int>>
findZeroInitOp(Value accUse, scf::ForOp forOp, bool &loopArgIsZero) {
  Value v = accUse;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    assert(arg.getOwner() == forOp.getBody());
    if (isConstantZeroTensor(forOp.getInitArgs()[arg.getArgNumber() - 1]))
      loopArgIsZero = true;
    v = forOp.getBody()->getTerminator()->getOperand(arg.getArgNumber() - 1);
  }

  auto defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  if (auto selOp = dyn_cast<arith::SelectOp>(defOp)) {
    if (!selOp.getCondition().getType().isInteger(1))
      return std::nullopt;
    if (isConstantZeroTensor(selOp.getTrueValue()) ||
        isConstantZeroTensor(selOp.getFalseValue())) {
      return std::make_pair(selOp, 0);
    }
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    unsigned resultIndex = cast<OpResult>(v).getResultNumber();
    Value thenVal = ifOp.thenYield()->getOperand(resultIndex);
    Value elseVal = ifOp.elseYield()->getOperand(resultIndex);
    if (isConstantZeroTensor(thenVal) || isConstantZeroTensor(elseVal)) {
      if (thenVal.getParentBlock()->getParentOp() == ifOp ||
          elseVal.getParentBlock()->getParentOp() == ifOp) {
        return std::nullopt;
      }
      return std::make_pair(ifOp, resultIndex);
    }
  }
  return std::nullopt;
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUOPTIMIZEACCUMULATORINIT
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUOptimizeAccumulatorInitPass
    : impl::TritonMUSAGPUOptimizeAccumulatorInitBase<
          TritonMUSAGPUOptimizeAccumulatorInitPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<Operation *> dotOps;
    mod.walk([&](Operation *op) {
      if (isMusaDotOp(op) && supportsAccumulatorInitOptimization(op))
        dotOps.push_back(op);
    });

    for (Operation *dotOp : dotOps) {
      auto forOp = dyn_cast<scf::ForOp>(dotOp->getParentOp());
      if (!forOp)
        continue;

      Location loc = dotOp->getLoc();
      IRRewriter rewriter(forOp);
      rewriter.setInsertionPoint(forOp);

      Value vTrue =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      Value vFalse =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));

      Value accUse = getAccumulatorValue(dotOp);
      if (!accUse)
        continue;

      Value useCValue = getUseCValue(dotOp);
      if (useCValue) {
        auto useCConst = tt::getBoolFromConstant(useCValue);
        if (!useCConst || !*useCConst)
          continue;
      }

      if (isConstantZeroTensor(accUse)) {
        setUseCValue(dotOp, vFalse);
        continue;
      }

      bool loopArgIsZero = false;
      std::optional<std::pair<Operation *, int>> zeroInitOp =
          findZeroInitOp(accUse, forOp, loopArgIsZero);
      if (!zeroInitOp && !loopArgIsZero)
        continue;

      Value loopArgFlagValue = loopArgIsZero ? vFalse : vTrue;
      forOp = addIterArgsToLoop(rewriter, forOp, {loopArgFlagValue});
      loopArgFlagValue =
          forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 1);

      if (zeroInitOp) {
        Value condition;
        Value oldValue;
        bool thenInitsToZero = false;
        if (auto selOp = dyn_cast<arith::SelectOp>(zeroInitOp->first)) {
          condition = selOp.getCondition();
          oldValue = isConstantZeroTensor(selOp.getTrueValue())
                         ? selOp.getFalseValue()
                         : selOp.getTrueValue();
          thenInitsToZero = isConstantZeroTensor(selOp.getTrueValue());
        } else {
          auto ifOp = cast<scf::IfOp>(zeroInitOp->first);
          unsigned resultIndex = zeroInitOp->second;
          condition = ifOp.getCondition();
          Value thenVal = ifOp.thenYield()->getOperand(resultIndex);
          Value elseVal = ifOp.elseYield()->getOperand(resultIndex);
          oldValue = isConstantZeroTensor(thenVal) ? elseVal : thenVal;
          thenInitsToZero = isConstantZeroTensor(thenVal);
        }

        rewriter.setInsertionPoint(zeroInitOp->first);
        bool zeroingBeforeDot = zeroInitOp->first->isBeforeInBlock(dotOp);
        Value prevFlagValue = zeroingBeforeDot ? loopArgFlagValue : vTrue;
        auto selectFlagOp = arith::SelectOp::create(
            rewriter, loc, condition, thenInitsToZero ? vFalse : prevFlagValue,
            thenInitsToZero ? prevFlagValue : vFalse);
        setUseCValue(dotOp, zeroingBeforeDot ? selectFlagOp : loopArgFlagValue);

        auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        forYield->insertOperands(forYield->getNumOperands(),
                                 {zeroingBeforeDot ? vTrue : selectFlagOp});

        if (auto selOp = dyn_cast<arith::SelectOp>(zeroInitOp->first)) {
          rewriter.setInsertionPoint(selOp);
          rewriter.replaceOp(selOp, oldValue);
        } else {
          auto ifOp = cast<scf::IfOp>(zeroInitOp->first);
          int resultIndex = zeroInitOp->second;
          auto zeroingYield =
              thenInitsToZero ? ifOp.thenYield() : ifOp.elseYield();
          zeroingYield.setOperand(resultIndex, oldValue);
        }
      } else if (loopArgIsZero) {
        setUseCValue(dotOp, loopArgFlagValue);
        auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        forYield->insertOperands(forYield->getNumOperands(), vTrue);
      }
    }
  }
};

} // namespace mlir
