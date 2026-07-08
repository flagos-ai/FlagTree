#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLELOWERWGMMA
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static constexpr unsigned kScfForControlOperands = 3;

static std::optional<unsigned> getForInitArgIndex(OpOperand &use) {
  auto forOp = dyn_cast<scf::ForOp>(use.getOwner());
  if (!forOp)
    return std::nullopt;
  unsigned operandNumber = use.getOperandNumber();
  if (operandNumber < kScfForControlOperands)
    return std::nullopt;
  return operandNumber - kScfForControlOperands;
}

static bool isForYield(OpOperand &use) {
  auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner());
  return yieldOp && isa<scf::ForOp>(yieldOp->getParentOp());
}

static bool isAsyncAccumulatorUse(OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<WGMMAOp>(user))
    return use.getOperandNumber() == 2;
  if (isa<WGMMAWaitOp>(user))
    return true;
  if (getForInitArgIndex(use))
    return true;
  return isForYield(use);
}

static LogicalResult verifyWGMMAUses(WGMMAOp op) {
  for (OpOperand &use : op.getD().getUses()) {
    Operation *user = use.getOwner();
    if (auto next = dyn_cast<WGMMAOp>(user)) {
      if (use.getOperandNumber() == 2)
        continue;
      return op.emitOpError("result may only feed the accumulator operand of "
                            "another tle.wgmma or tle.wgmma_wait");
    }
    if (isa<WGMMAWaitOp>(user))
      continue;
    if (getForInitArgIndex(use))
      continue;
    return op.emitOpError("async result must be consumed by tle.wgmma_wait "
                          "before ordinary tensor use");
  }
  return success();
}

static RankedTensorType getMMAType(WGMMAOp op) {
  auto context = op.getContext();
  auto accType = cast<RankedTensorType>(op.getC().getType());
  auto aElemType =
      cast<ttg::TensorOrMemDesc>(op.getA().getType()).getElementType();
  std::optional<int> maybeNumWarps =
      ttg::maybeLookupNumWarps(op.getOperation());
  if (!maybeNumWarps) {
    op.emitOpError("requires a contextual `ttg.num-warps` to select the "
                   "WGMMA accumulator layout");
    return {};
  }

  unsigned numWarps = *maybeNumWarps;
  SmallVector<int64_t> retShapePerCTA =
      accType.getEncoding() ? ttg::getShapePerCTA(accType)
                            : SmallVector<int64_t>(accType.getShape());
  SmallVector<unsigned, 3> instrShape =
      mmaVersionToInstrShape(3, retShapePerCTA, aElemType, numWarps);
  SmallVector<unsigned, 2> warpsPerCTA = {numWarps, 1};
  SmallVector<unsigned, 2> CTAsPerCGA = {1, 1};
  SmallVector<unsigned, 2> CTASplitNum = {1, 1};
  SmallVector<unsigned, 2> CTAOrder = {1, 0};
  auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(context, CTAsPerCGA,
                                                         CTASplitNum, CTAOrder);
  auto mmaEncoding = ttg::NvidiaMmaEncodingAttr::get(context, 3, 0, warpsPerCTA,
                                                     CTALayout, instrShape);
  return RankedTensorType::get(accType.getShape(), accType.getElementType(),
                               mmaEncoding);
}

static bool isMMAEncoded(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return false;
  Attribute encoding = type.getEncoding();
  return encoding && isa<ttg::NvidiaMmaEncodingAttr>(encoding);
}

static Value lookupEncodedAccumulator(Value value,
                                      DenseMap<Value, Value> &encodedAccs) {
  if (Value mapped = encodedAccs.lookup(value))
    return mapped;
  if (isMMAEncoded(value))
    return value;
  return {};
}

static void convertLoopCarriedAccumulator(OpOperand &use, Value encodedInit,
                                          DenseMap<Value, Value> &encodedAccs) {
  std::optional<unsigned> maybeIndex = getForInitArgIndex(use);
  if (!maybeIndex)
    return;

  auto forOp = cast<scf::ForOp>(use.getOwner());
  unsigned initIndex = *maybeIndex;
  Type encodedType = encodedInit.getType();
  forOp->setOperand(use.getOperandNumber(), encodedInit);

  BlockArgument regionArg =
      forOp.getBody()->getArgument(forOp.getNumInductionVars() + initIndex);
  regionArg.setType(encodedType);
  forOp.getResult(initIndex).setType(encodedType);

  encodedAccs[regionArg] = regionArg;
  encodedAccs[forOp.getResult(initIndex)] = forOp.getResult(initIndex);
}

static Value materializeMMAAccumulator(OpBuilder &builder, WGMMAOp op,
                                       RankedTensorType mmaType,
                                       DenseMap<Value, Value> &encodedAccs) {
  Value acc = op.getC();
  if (Value mapped = lookupEncodedAccumulator(acc, encodedAccs))
    return mapped;

  auto accType = cast<RankedTensorType>(acc.getType());
  if (accType.getEncoding() == mmaType.getEncoding())
    return acc;
  return ttg::ConvertLayoutOp::create(builder, op.getLoc(), mmaType, acc);
}

static Value materializeAOperand(OpBuilder &builder, WGMMAOp op,
                                 RankedTensorType mmaType) {
  Value a = op.getA();
  if (isa<ttg::MemDescType>(a.getType()))
    return a;

  auto aType = cast<RankedTensorType>(a.getType());
  Attribute dotEncoding = ttg::DotOperandEncodingAttr::get(
      op.getContext(), /*opIdx=*/0, mmaType.getEncoding(),
      aType.getElementType());
  auto dotType = aType.cloneWithEncoding(dotEncoding);
  if (aType == dotType)
    return a;
  return ttg::ConvertLayoutOp::create(builder, op.getLoc(), dotType, a);
}

struct TritonTleLowerWGMMAPass
    : public impl::TritonTleLowerWGMMABase<TritonTleLowerWGMMAPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool failed = false;

    module.walk([&](triton::FuncOp func) {
      if (failed)
        return;

      SmallVector<Operation *> worklist;
      SmallVector<WGMMAOp> wgmmas;
      func.walk([&](Operation *op) {
        if (isa<WGMMAOp, WGMMAWaitOp>(op))
          worklist.push_back(op);
        if (auto wgmma = dyn_cast<WGMMAOp>(op))
          wgmmas.push_back(wgmma);
      });

      for (WGMMAOp op : wgmmas) {
        if (mlir::failed(verifyWGMMAUses(op))) {
          failed = true;
          return;
        }
      }

      DenseMap<Value, Value> encodedAccs;
      for (Operation *op : worklist) {
        OpBuilder builder(op);
        if (auto wgmma = dyn_cast<WGMMAOp>(op)) {
          RankedTensorType mmaType = getMMAType(wgmma);
          if (!mmaType) {
            failed = true;
            return;
          }
          Value acc =
              materializeMMAAccumulator(builder, wgmma, mmaType, encodedAccs);
          Value a = materializeAOperand(builder, wgmma, mmaType);
          if (!acc) {
            failed = true;
            return;
          }
          auto nativeDot = ttng::WarpGroupDotOp::create(
              builder, wgmma.getLoc(), acc.getType(), a, wgmma.getB(), acc,
              Value(), wgmma.getInputPrecision(), wgmma.getMaxNumImpreciseAcc(),
              wgmma.getIsAsync());
          encodedAccs[wgmma.getD()] = nativeDot.getD();
          for (OpOperand &use :
               llvm::make_early_inc_range(wgmma.getD().getUses()))
            convertLoopCarriedAccumulator(use, nativeDot.getD(), encodedAccs);
          continue;
        }

        auto wait = cast<WGMMAWaitOp>(op);
        Value encodedInput =
            lookupEncodedAccumulator(wait.getInput(), encodedAccs);
        if (!encodedInput) {
          wait.emitOpError("input must be the async result of tle.wgmma");
          failed = true;
          return;
        }

        SmallVector<Value, 1> waitInputs{encodedInput};
        auto nativeWait = ttng::WarpGroupDotWaitOp::create(
            builder, wait.getLoc(), waitInputs, wait.getPendings());
        Value waited = nativeWait.getResult(0);
        if (wait.getPendings() > 0) {
          // Rewrite every async accumulator use directly before erasing this
          // wait. Keeping wait.getOutput() in encodedAccs would leave a map
          // entry keyed by a soon-to-be-dangling Value; later rewrites may then
          // accidentally pick an accumulator from another isolated region.
          Value released;
          for (OpOperand &use :
               llvm::make_early_inc_range(wait.getOutput().getUses())) {
            if (isAsyncAccumulatorUse(use)) {
              if (getForInitArgIndex(use))
                convertLoopCarriedAccumulator(use, waited, encodedAccs);
              else
                use.set(waited);
              continue;
            }

            if (!released)
              released = ttg::ConvertLayoutOp::create(
                  builder, wait.getLoc(), wait.getOutput().getType(), waited);
            use.set(released);
          }
          wait.erase();
          continue;
        }

        Value released = ttg::ConvertLayoutOp::create(
            builder, wait.getLoc(), wait.getOutput().getType(), waited);
        wait.getOutput().replaceAllUsesWith(released);
        wait.erase();
      }

      for (WGMMAOp op : llvm::reverse(wgmmas))
        op.erase();
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
