#include "Dialect/TritonILUVATARGPU/Utility/CommonUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::ILUVATAR {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

SmallVector<unsigned> getShapePerCTATile(RankedTensorType tensorTy) {
  auto llEnc = triton::gpu::toLinearEncoding(tensorTy);
  auto sizePerThread = llEnc.getSizePerThread();
  auto threadsPerWarp = llEnc.getThreadsPerWarp();
  auto warpsPerCTA = llEnc.getWarpsPerCTA();
  SmallVector<unsigned> shape;
  for (auto [size, thread, warp] :
       llvm::zip(sizePerThread, threadsPerWarp, warpsPerCTA)) {
    shape.push_back(size * thread * warp);
  }
  return shape;
}

ElemLocationKey getElemCoordinatesFromRegisters(triton::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx) {
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation = {
      {kReg, static_cast<int32_t>(regId)},
      {kLane, 0},
      {kWarp, 0},
      {kBlock, 0},
  };

  return ll.apply(hardwareLocation);
}

std::optional<int> getRegFromCoordinates(triton::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx) {
  auto dims = ll.pseudoinvert().apply(coordinates);
  StringAttr kReg = StringAttr::get(ctx, "register");
  assert(dims[0].first == kReg && "First dimension must be 'register'");

  int regId = dims[0].second; // "register"
  if (dims[1].second != 0 || dims[2].second != 0 || dims[3].second != 0)
    return std::nullopt;
  return regId;
}
} // namespace mlir::triton::ILUVATAR

namespace tt = mlir::triton;

namespace mlir::LLVM::ILUVATAR {
namespace {

void getFwdSliceImpl(Operation *op, SetVector<Operation *> *forwardSlice,
                     Value targetValue);
void getBwdSliceImpl(Operation *op, SetVector<Operation *> *backwardSlice,
                     bool omitBlockArguments);

static void processTerminator(Operation *terminator, Value targetValue,
                              SetVector<Operation *> *forwardSlice) {
  Block *block = terminator->getBlock();
  if (!block)
    return;

  Region *region = block->getParent();
  if (!region)
    return;

  Operation *outerOp = region->getParentOp();
  if (!outerOp)
    return;

  for (auto [idx, operand] : llvm::enumerate(terminator->getOperands())) {
    if (operand != targetValue)
      continue;
    if (idx >= outerOp->getNumResults())
      continue;

    Value outerResult = outerOp->getResult(idx);
    for (Operation *user : outerResult.getUsers()) {
      if (!forwardSlice->count(user))
        getFwdSliceImpl(user, forwardSlice, outerResult);
    }
  }
}

static void trackBlockUsers(Block &block, Value targetValue,
                            SetVector<Operation *> *forwardSlice) {
  for (auto blockArg : block.getArguments()) {
    if (blockArg == targetValue) {
      for (Operation *user : blockArg.getUsers())
        if (!forwardSlice->count(user))
          getFwdSliceImpl(user, forwardSlice, targetValue);
      return;
    }
  }

  for (Operation &op : block) {
    for (Value operand : op.getOperands()) {
      if (operand == targetValue) {
        if (!forwardSlice->count(&op))
          getFwdSliceImpl(&op, forwardSlice, targetValue);
        break;
      }
    }
  }
}

void getFwdSliceImpl(Operation *op, SetVector<Operation *> *forwardSlice,
                     Value targetValue) {
  if (!op || forwardSlice->count(op))
    return;

  forwardSlice->insert(op);

  if (op->hasTrait<OpTrait::IsTerminator>()) {
    processTerminator(op, targetValue, forwardSlice);
    return;
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (auto en : llvm::enumerate(forOp.getInitArgs())) {
      if (en.value() != targetValue)
        continue;
      for (Operation *user : op->getResult(en.index()).getUsers()) {
        if (!forwardSlice->count(user))
          getFwdSliceImpl(user, forwardSlice, nullptr);
      }
      Block *body = forOp.getBody();
      for (Operation *user : body->getArgument(en.index() + 1).getUsers()) {
        if (!forwardSlice->count(user))
          getFwdSliceImpl(user, forwardSlice, nullptr);
      }
    }
  } else {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        trackBlockUsers(block, targetValue, forwardSlice);

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!forwardSlice->count(user))
          getFwdSliceImpl(user, forwardSlice, result);
      }
    }
  }
}

static void getFwdSliceOp(Operation *op, SetVector<Operation *> *forwardSlice) {
  getFwdSliceImpl(op, forwardSlice, nullptr);
}

static void visitRegionResult(Operation *op, unsigned resultIdx,
                              SetVector<Operation *> *backwardSlice,
                              bool omitBlockArguments) {
  for (auto &region : op->getRegions()) {
    if (region.empty())
      continue;
    Operation *terminator = region.front().getTerminator();
    if (!terminator || resultIdx >= terminator->getNumOperands())
      continue;

    Value yieldOperand = terminator->getOperand(resultIdx);
    if (auto *definingOp = yieldOperand.getDefiningOp()) {
      getBwdSliceImpl(definingOp, backwardSlice, omitBlockArguments);
    } else if (auto blockArg = dyn_cast<BlockArgument>(yieldOperand)) {
      if (!omitBlockArguments) {
        Operation *parentOp = blockArg.getOwner()->getParentOp();
        if (parentOp && !backwardSlice->count(parentOp))
          getBwdSliceImpl(parentOp, backwardSlice, omitBlockArguments);
      }
    }
  }
}

void getBwdSliceImpl(Operation *op, SetVector<Operation *> *backwardSlice,
                     bool omitBlockArguments) {
  if (!op || backwardSlice->count(op))
    return;

  for (Value operand : op->getOperands()) {
    if (auto *definingOp = operand.getDefiningOp()) {
      if (auto result = dyn_cast<OpResult>(operand)) {
        Operation *parentOp = result.getOwner();
        unsigned resultIdx = result.getResultNumber();
        if (!parentOp->getRegions().empty()) {
          visitRegionResult(parentOp, resultIdx, backwardSlice,
                            omitBlockArguments);
          continue;
        }
      }
      getBwdSliceImpl(definingOp, backwardSlice, omitBlockArguments);
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      if (!omitBlockArguments) {
        Operation *parentOp = blockArg.getOwner()->getParentOp();
        if (parentOp && !backwardSlice->count(parentOp))
          getBwdSliceImpl(parentOp, backwardSlice, omitBlockArguments);
      }
    }
  }

  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator)
        continue;
      for (Value operand : terminator->getOperands()) {
        if (auto *definingOp = operand.getDefiningOp()) {
          getBwdSliceImpl(definingOp, backwardSlice, omitBlockArguments);
        } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (!omitBlockArguments) {
            Operation *parentOp = blockArg.getOwner()->getParentOp();
            if (parentOp && !backwardSlice->count(parentOp))
              getBwdSliceImpl(parentOp, backwardSlice, omitBlockArguments);
          }
        }
      }
    }
  }

  backwardSlice->insert(op);
}

static void getBwdSlice(Operation *op, SetVector<Operation *> *backwardSlice,
                        bool omitBlockArguments) {
  getBwdSliceImpl(op, backwardSlice, omitBlockArguments);
}

} // namespace

void analyzeDotChain(tt::DotOpInterface dotOp, DotChainInfo &info) {
  info = {};

  SetVector<Operation *> fwdSlices;
  getFwdSliceOp(dotOp, &fwdSlices);
  for (Operation *op : fwdSlices) {
    if (auto dOp = dyn_cast<tt::DotOpInterface>(op)) {
      if (dOp == dotOp)
        continue;
      Operation *opA = dOp.getA().getDefiningOp();
      if (opA && fwdSlices.contains(opA)) {
        info.useAsA = true;
        info.isHeadDot = true;
      }
      Operation *opB = dOp.getB().getDefiningOp();
      if (opB && fwdSlices.contains(opB)) {
        info.useAsB = true;
        info.isHeadDot = true;
      }
    }
  }

  auto traceOperand = [&](Value operand, bool asA) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      return;
    SetVector<Operation *> bwdSlices;
    getBwdSlice(defOp, &bwdSlices, /*omitBlockArguments=*/true);
    if (llvm::any_of(bwdSlices, [](Operation *op) {
          return isa<tt::DotOpInterface>(op);
        })) {
      if (asA)
        info.defAsA = true;
      else
        info.defAsB = true;
      info.isTailDot = true;
    }
  };

  traceOperand(dotOp.getA(), /*asA=*/true);
  traceOperand(dotOp.getB(), /*asA=*/false);
}

bool isChainDotHead(tt::DotOpInterface dotOp, unsigned opIdx) {
  DotChainInfo info;
  analyzeDotChain(dotOp, info);
  if (!info.isHeadDot)
    return false;
  return opIdx == 0 ? info.useAsA : info.useAsB;
}

bool isChainDotTail(tt::DotOpInterface dotOp) {
  DotChainInfo info;
  analyzeDotChain(dotOp, info);
  return info.isTailDot;
}

} // namespace mlir::LLVM::ILUVATAR
