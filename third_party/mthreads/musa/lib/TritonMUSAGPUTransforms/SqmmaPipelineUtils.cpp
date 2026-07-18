#include "SqmmaPipelineUtils.h"

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::musa::pipeline {
namespace {

using ProperlyAsyncDots = llvm::MapVector<Operation *, int /*iterArgIdx*/>;

bool isInCurrentPipelineLoop(Operation *op, scf::ForOp loop) {
  return op->getParentOfType<scf::ForOp>() == loop;
}

Value getTransparentSqmmaOperandSource(Operation *op) {
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
    return cvt.getSrc();
  if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(op))
    return localLoad.getSrc();
  if (auto index = dyn_cast<ttg::MemDescIndexOp>(op))
    return index.getSrc();
  if (auto subslice = dyn_cast<ttg::MemDescSubsliceOp>(op))
    return subslice.getSrc();
  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op))
    return trans.getSrc();
  if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op))
    return reshape.getSrc();
  if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(op))
    return reinterpret.getSrc();
  if (isNoop(op) && op->getNumOperands() == 1 && op->getNumResults() == 1)
    return op->getOperand(0);
  return Value();
}

struct ResolvedSqmmaValue {
  Value root;
  bool loopCarryCycle = false;
};

ResolvedSqmmaValue resolveSqmmaValueRoot(Value value, scf::ForOp loop) {
  llvm::SmallPtrSet<void *, 16> visited;
  while (value) {
    if (loop.isDefinedOutsideOfLoop(value))
      return {value, false};

    if (!visited.insert(value.getAsOpaquePointer()).second)
      return {value, true};

    if (Operation *defOp = value.getDefiningOp()) {
      if (Value source = getTransparentSqmmaOperandSource(defOp)) {
        value = source;
        continue;
      }
      return {value, false};
    }

    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg || blockArg.getOwner() != loop.getBody())
      return {value, false};

    OpOperand *yielded = loop.getTiedLoopYieldedValue(blockArg);
    if (!yielded)
      return {value, false};
    value = yielded->get();
  }
  return {};
}

bool sqmmaOperandCanBeProperlyAsync(Value operand, scf::ForOp loop) {
  ResolvedSqmmaValue resolved = resolveSqmmaValueRoot(operand, loop);
  return resolved.loopCarryCycle ||
         (resolved.root && loop.isDefinedOutsideOfLoop(resolved.root));
}

Value getWaitThreadableMemDesc(Value operand, scf::ForOp loop) {
  ResolvedSqmmaValue resolved = resolveSqmmaValueRoot(operand, loop);
  if (!resolved.root || !isa<ttg::MemDescType>(resolved.root.getType()))
    return Value();
  return resolved.root;
}

bool hasLoopLocalSharedReuse(Value operand, scf::ForOp loop) {
  ResolvedSqmmaValue resolved = resolveSqmmaValueRoot(operand, loop);
  return resolved.root && isa<ttg::MemDescType>(resolved.root.getType()) &&
         !resolved.loopCarryCycle &&
         !loop.isDefinedOutsideOfLoop(resolved.root);
}

void collectDependentSqmmaDots(Value value, scf::ForOp loop,
                               llvm::SetVector<triton::musa::SquadDotOp> &dots,
                               llvm::SmallPtrSetImpl<void *> &visited);

void collectIfResultSqmmaDots(scf::IfOp ifOp, unsigned resultIdx,
                              scf::ForOp loop,
                              llvm::SetVector<triton::musa::SquadDotOp> &dots,
                              llvm::SmallPtrSetImpl<void *> &visited) {
  auto collectFromRegion = [&](Region &region) {
    auto yieldOp = dyn_cast<scf::YieldOp>(region.front().getTerminator());
    if (!yieldOp || resultIdx >= yieldOp.getNumOperands())
      return;
    collectDependentSqmmaDots(yieldOp.getOperand(resultIdx), loop, dots,
                              visited);
  };
  collectFromRegion(ifOp.getThenRegion());
  if (ifOp.elseBlock())
    collectFromRegion(ifOp.getElseRegion());
}

void collectDependentSqmmaDots(Value value, scf::ForOp loop,
                               llvm::SetVector<triton::musa::SquadDotOp> &dots,
                               llvm::SmallPtrSetImpl<void *> &visited) {
  if (!value || !visited.insert(value.getAsOpaquePointer()).second)
    return;

  if (loop.isDefinedOutsideOfLoop(value))
    return;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == loop.getBody()) {
      if (OpOperand *yielded = loop.getTiedLoopYieldedValue(blockArg))
        collectDependentSqmmaDots(yielded->get(), loop, dots, visited);
    }
    return;
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return;

  Operation *defOp = result.getOwner();
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(defOp)) {
    if (!isInCurrentPipelineLoop(sqmma, loop))
      return;
    dots.insert(sqmma);
    collectDependentSqmmaDots(sqmma.getC(), loop, dots, visited);
    return;
  }

  if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(defOp)) {
    unsigned idx = result.getResultNumber();
    if (idx < wait.getInputs().size())
      collectDependentSqmmaDots(wait.getInputs()[idx], loop, dots, visited);
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    collectIfResultSqmmaDots(ifOp, result.getResultNumber(), loop, dots,
                             visited);
    return;
  }

  if (auto forResult = dyn_cast<scf::ForOp>(defOp)) {
    if (forResult == loop &&
        result.getResultNumber() < loop.getNumRegionIterArgs()) {
      collectDependentSqmmaDots(loop.getRegionIterArg(result.getResultNumber()),
                                loop, dots, visited);
    }
    return;
  }

  if (Value source = getTransparentSqmmaOperandSource(defOp)) {
    collectDependentSqmmaDots(source, loop, dots, visited);
    return;
  }
}

void collectSqmmaMemDescDependencies(ArrayRef<Value> values,
                                     Operation *waitAnchor, scf::ForOp loop,
                                     llvm::SetVector<Value> &memDescs) {
  llvm::SetVector<triton::musa::SquadDotOp> dots;
  llvm::SmallPtrSet<void *, 32> visited;
  for (Value value : values)
    collectDependentSqmmaDots(value, loop, dots, visited);

  Operation *domRoot = waitAnchor->getParentOfType<tt::FuncOp>();
  if (!domRoot)
    domRoot = waitAnchor->getParentOfType<ModuleOp>();
  DominanceInfo domInfo(domRoot ? domRoot : waitAnchor->getParentOp());

  for (triton::musa::SquadDotOp sqmma : dots) {
    for (Value operand : {sqmma.getA(), sqmma.getB()}) {
      Value memDesc = getWaitThreadableMemDesc(operand, loop);
      if (!memDesc || !domInfo.properlyDominates(memDesc, waitAnchor))
        continue;
      memDescs.insert(memDesc);
    }
  }
}

triton::musa::SquadDotWaitOp
threadValuesThroughWait(triton::musa::SquadDotWaitOp wait,
                        ArrayRef<Value> values, scf::ForOp loop) {
  IRRewriter rewriter(wait.getContext());
  rewriter.setInsertionPoint(wait);

  const unsigned origNumOperands = wait.getNumOperands();
  llvm::SetVector<Value> operands(wait.getInputs().begin(),
                                  wait.getInputs().end());
  operands.insert(values.begin(), values.end());

  llvm::SetVector<Value> memDescs;
  SmallVector<Value> dependencyRoots(operands.begin(), operands.end());
  collectSqmmaMemDescDependencies(dependencyRoots, wait, loop, memDescs);
  operands.insert(memDescs.begin(), memDescs.end());

  if (operands.size() == origNumOperands)
    return wait;

  SmallVector<Value> newOperands(operands.begin(), operands.end());
  auto newWait = triton::musa::SquadDotWaitOp::create(rewriter, wait.getLoc(),
                                                      newOperands);
  newWait->setAttrs(wait->getAttrs());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto *topLevel =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return topLevel && newWait->isBeforeInBlock(topLevel);
  };

  for (unsigned idx = 0; idx < origNumOperands; ++idx) {
    Value oldResult = wait.getResult(idx);
    if (!isa<ttg::MemDescType>(oldResult.getType()))
      oldResult.replaceAllUsesWith(newWait.getResult(idx));
  }
  for (unsigned idx = origNumOperands; idx < newOperands.size(); ++idx) {
    Value operand = newWait.getOperand(idx);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(idx), dominatedByNewWait);
  }

  rewriter.eraseOp(wait);
  return newWait;
}

bool isTransitivelySqmmaCUse(OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<triton::musa::SquadDotOp>(user))
    return use.getOperandNumber() == 2;
  if (isNoop(user) && user->getNumResults() == 1)
    return llvm::all_of(user->getResult(0).getUses(), isTransitivelySqmmaCUse);
  return false;
}

std::optional<int> dotCanBeProperlyAsync(triton::musa::SquadDotOp sqmma,
                                         scf::ForOp loop) {
  if (!sqmmaOperandCanBeProperlyAsync(sqmma.getA(), loop) ||
      !sqmmaOperandCanBeProperlyAsync(sqmma.getB(), loop)) {
    return std::nullopt;
  }

  if (auto cArg = dyn_cast<BlockArgument>(sqmma.getC())) {
    if (cArg.getOwner() == loop.getBody() && cArg.getArgNumber() > 0) {
      if (OpOperand *yielded = loop.getTiedLoopYieldedValue(cArg)) {
        if (yielded->get() == sqmma.getResult() &&
            llvm::all_of(cArg.getUses(), isTransitivelySqmmaCUse)) {
          return static_cast<int>(cArg.getArgNumber() - 1);
        }
      }
    }
  }

  SmallVector<std::pair<int, Value>> yieldedIterArgs;
  auto recordYieldedIterArg = [&](int operandIdx) -> bool {
    if (operandIdx >= loop.getNumRegionIterArgs())
      return false;
    if (llvm::none_of(yieldedIterArgs, [&](const auto &entry) {
          return entry.first == operandIdx;
        })) {
      yieldedIterArgs.push_back(
          {operandIdx, loop.getRegionIterArg(operandIdx)});
    }
    return true;
  };

  SmallVector<std::pair<Operation *, int>> queue;
  for (OpOperand &use : sqmma->getUses())
    queue.push_back({use.getOwner(), static_cast<int>(use.getOperandNumber())});

  while (!queue.empty()) {
    auto [user, operandIdx] = queue.pop_back_val();
    if (user->getParentOp() == loop) {
      if (isNoop(user) && user->getNumResults() == 1) {
        for (OpOperand &use : user->getResult(0).getUses())
          queue.push_back(
              {use.getOwner(), static_cast<int>(use.getOperandNumber())});
        continue;
      }
      if (isa<scf::YieldOp>(user)) {
        if (!recordYieldedIterArg(operandIdx))
          return std::nullopt;
        continue;
      }
      return std::nullopt;
    }

    auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp());
    if (!ifOp)
      return std::nullopt;

    if (isa<scf::YieldOp>(user)) {
      for (OpOperand &use : ifOp.getResult(operandIdx).getUses())
        queue.push_back(
            {use.getOwner(), static_cast<int>(use.getOperandNumber())});
    }
  }

  if (yieldedIterArgs.empty())
    return std::nullopt;

  auto selectIterArg =
      [&](auto &&predicate) -> std::optional<std::pair<int, Value>> {
    std::optional<std::pair<int, Value>> selected;
    for (const auto &entry : yieldedIterArgs) {
      if (!predicate(entry.second))
        continue;
      if (selected)
        return std::nullopt;
      selected = entry;
    }
    return selected;
  };

  auto otherYieldedIterArgsAreLoopDead = [&](int selectedIdx) {
    return llvm::all_of(yieldedIterArgs, [&](const auto &entry) {
      return entry.first == selectedIdx || entry.second.use_empty();
    });
  };

  if (auto selected = selectIterArg([&](Value candidate) {
        return candidate == sqmma.getC() &&
               llvm::all_of(candidate.getUses(), isTransitivelySqmmaCUse);
      });
      selected && otherYieldedIterArgsAreLoopDead(selected->first)) {
    return selected->first;
  }

  if (auto selected = selectIterArg([&](Value candidate) {
        return llvm::all_of(candidate.getUses(), isTransitivelySqmmaCUse);
      });
      selected && otherYieldedIterArgsAreLoopDead(selected->first)) {
    return selected->first;
  }

  auto waitOps = loop.getBody()->getOps<triton::musa::SquadDotWaitOp>();
  auto firstWait =
      llvm::find_if(waitOps, [](triton::musa::SquadDotWaitOp) { return true; });
  auto iterArgUsersAreAfterFirstWait = [&](Value candidate) {
    return llvm::all_of(candidate.getUsers(), [&](Operation *user) {
      assert(loop->isAncestor(user));
      while (user->getParentOp() != loop)
        user = user->getParentOp();
      return (*firstWait)->isBeforeInBlock(user);
    });
  };
  if (firstWait != waitOps.end()) {
    auto selected = selectIterArg(iterArgUsersAreAfterFirstWait);
    if (selected && otherYieldedIterArgsAreLoopDead(selected->first)) {
      threadValuesThroughWait(*firstWait, {selected->second}, loop);
      return selected->first;
    }
  }

  return std::nullopt;
}

triton::musa::SquadDotWaitOp getOrCreateWaitBefore(Operation *op) {
  if (auto wait =
          dyn_cast_or_null<triton::musa::SquadDotWaitOp>(op->getPrevNode()))
    return wait;

  OpBuilder builder(op);
  return triton::musa::SquadDotWaitOp::create(builder, op->getLoc(),
                                              ArrayRef<Value>{});
}

triton::musa::SquadDotWaitOp getOrCreateWaitAfter(Operation *op) {
  if (auto wait =
          dyn_cast_or_null<triton::musa::SquadDotWaitOp>(op->getNextNode()))
    return wait;

  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  return triton::musa::SquadDotWaitOp::create(builder, op->getLoc(),
                                              ArrayRef<Value>{});
}

void insertAsyncSqmmaWaitsInLoop(scf::ForOp loop,
                                 const ProperlyAsyncDots &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  for (auto [asyncDotOp, iterArgIdx] : properlyAsyncDots) {
    (void)iterArgIdx;
    auto asyncDot = cast<triton::musa::SquadDotOp>(asyncDotOp);
    DenseMap<Block *, SmallVector<OpOperand *>> blockToUses;
    for (OpOperand &use : asyncDot->getUses()) {
      if (isa<scf::YieldOp>(use.getOwner()))
        continue;
      blockToUses[use.getOwner()->getBlock()].push_back(&use);
    }

    for (auto &entry : blockToUses) {
      auto &uses = entry.second;
      std::sort(uses.begin(), uses.end(), [](OpOperand *lhs, OpOperand *rhs) {
        return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
      });

      auto firstUse =
          std::find_if_not(uses.begin(), uses.end(), [](OpOperand *use) {
            return isa<triton::musa::SquadDotOp>(use->getOwner()) &&
                   use->getOperandNumber() == 2;
          });
      if (firstUse == uses.end())
        continue;

      Operation *firstConsumer = (*firstUse)->getOwner();
      SmallVector<Value> waitOperands;
      for (auto useIt = firstUse; useIt != uses.end(); ++useIt)
        waitOperands.push_back((*useIt)->get());

      auto wait = getOrCreateWaitBefore(firstConsumer);
      threadValuesThroughWait(wait, waitOperands, loop);
    }
  }
}

void insertFinalWaitAfterLoop(scf::ForOp loop,
                              const ProperlyAsyncDots &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  OpBuilder builder(loop);
  builder.setInsertionPointAfter(loop);
  auto wait = triton::musa::SquadDotWaitOp::create(builder, loop.getLoc(),
                                                   ArrayRef<Value>{});

  SmallVector<Value> waitOperands;
  waitOperands.reserve(properlyAsyncDots.size());
  for (auto [asyncDotOp, iterArgIdx] : properlyAsyncDots)
    waitOperands.push_back(loop.getResult(iterArgIdx));
  threadValuesThroughWait(wait, waitOperands, loop);
}

SmallVector<triton::musa::SquadDotOp>
collectPipelineSqmmaDots(scf::ForOp loop) {
  SmallVector<triton::musa::SquadDotOp> dots;
  loop.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
      dots.push_back(sqmma);
      return WalkResult::advance();
    }
    if (isa<scf::ForOp>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });
  return dots;
}

} // namespace

void pipelineSqmma(ModuleOp moduleOp, unsigned numStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp loop : loops) {
    if (tt::getNumStagesOrDefault(loop, numStages) < 1)
      continue;

    SmallVector<triton::musa::SquadDotOp> sqmmaDots =
        collectPipelineSqmmaDots(loop);
    if (sqmmaDots.empty())
      continue;

    ProperlyAsyncDots properlyAsyncDots;
    for (triton::musa::SquadDotOp sqmma : sqmmaDots) {
      sqmma->setAttr("isAsync", BoolAttr::get(moduleOp.getContext(), true));
      if (auto iterArgIdx = dotCanBeProperlyAsync(sqmma, loop)) {
        properlyAsyncDots[sqmma] = *iterArgIdx;
        continue;
      }

      auto wait = getOrCreateWaitAfter(sqmma);
      (void)threadValuesThroughWait(wait, {sqmma.getResult()}, loop);
    }

    insertAsyncSqmmaWaitsInLoop(loop, properlyAsyncDots);
    insertFinalWaitAfterLoop(loop, properlyAsyncDots);
  }
}

} // namespace mlir::triton::musa::pipeline
