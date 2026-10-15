#include "TritonILUVATARGPUTransforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <functional>
#include <limits>

// This pass mirrors AMD's UpdateAsyncWaitCount for Iluvatar SME:
// ttg.async_wait.num is the number of outstanding ttg.async_commit_groups.
// The pass converts that into the number of outstanding llvm.bi.sme.load
// intrinsics (G2S entries) so AsyncWaitOpConversion can encode G2S_CNT.
// Never overestimate the wait count; underestimation only waits more.

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONILUVATARGPUUPDATEASYNCWAITCOUNT
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kIluvatarG2SWaitAttrName =
    "ttg.iluvatar.g2sWaitCnt";

// Local copies of AMD's deduceMin helpers (anonymous namespace avoids ODR
// clashes with third_party/amd/.../Utility.cpp when both link into libtriton).
namespace deduceMin {
int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc);

int deduceMinCountBetweeOps(Operation *beginOp, Operation *endOp,
                            const std::function<int(Operation *)> &countFunc) {
  assert(beginOp && endOp);
  assert(beginOp == endOp || beginOp->isBeforeInBlock(endOp));
  int count = 0;
  for (auto op = beginOp; op != endOp; op = op->getNextNode()) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      assert(!ifOp.getThenRegion().empty() && !ifOp.getElseRegion().empty());
      auto minThen =
          deduceMinCountInBlock(ifOp.getThenRegion().front(), countFunc);
      auto minElse =
          deduceMinCountInBlock(ifOp.getElseRegion().front(), countFunc);
      count += std::min(minThen, minElse);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      if (std::optional<APInt> tripCount = forOp.getStaticTripCount()) {
        uint64_t tcVal = 0;
        if (forOp.getUnsignedCmp() && tripCount->ugt(0))
          tcVal = tripCount->getZExtValue();
        else if (!forOp.getUnsignedCmp() && tripCount->sgt(0))
          tcVal = tripCount->getSExtValue();
        if (tcVal > 0)
          count += tcVal * deduceMinCountInBlock(*forOp.getBody(), countFunc);
      }
    } else {
      count += countFunc(op);
    }
  }
  return count;
}

int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc) {
  if (block.empty())
    return 0;
  return deduceMinCountBetweeOps(&block.front(), &block.back(), countFunc);
}
} // namespace deduceMin

int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             const std::function<int(Operation *)> &countFunc,
                             int pathSum, int foundMin) {
  using namespace deduceMin;
  while (consumerOp->getParentRegion() != defValue.getParentRegion()) {
    pathSum += deduceMin::deduceMinCountBetweeOps(
        &consumerOp->getBlock()->front(), consumerOp, countFunc);
    consumerOp = consumerOp->getParentOp();
  }

  if (Operation *defOp = defValue.getDefiningOp()) {
    pathSum +=
        deduceMinCountBetweeOps(defOp->getNextNode(), consumerOp, countFunc);
    foundMin = std::min(foundMin, pathSum);
    return foundMin;
  }
  if (auto arg = mlir::dyn_cast<BlockArgument>(defValue)) {
    Block *block = arg.getOwner();
    auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

    if (!forOp || forOp.getBody()->empty()) {
      return 0;
    }

    Operation *firstOpInLoop = &*forOp.getBody()->begin();
    pathSum += deduceMinCountBetweeOps(firstOpInLoop, consumerOp, countFunc);

    if (pathSum >= foundMin)
      return foundMin;

    Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
    int countLoopInit = deduceMinCountOnDefChain(incomingVal, forOp, countFunc,
                                                 pathSum, foundMin);

    Operation *yieldOp = block->getTerminator();
    Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
    int countPreviousIter = deduceMinCountOnDefChain(
        prevVal, yieldOp, countFunc, pathSum, foundMin);

    return std::min(std::min(countLoopInit, countPreviousIter), foundMin);
  }

  return 0;
}

int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             llvm::function_ref<int(Operation *)> countFunc) {
  return deduceMinCountOnDefChain(defValue, consumerOp, countFunc, 0,
                                  std::numeric_limits<int>::max());
}

// Number of SME hardware loads emitted per active warp for one
// async_copy_global_to_local. Matches emitIluvatarSmeTileLoads's tile loops.
int getIluvatarSmeG2SCost(ttg::AsyncCopyGlobalToLocalOp op) {
  if (!op.isIluvatarSmeAsyncCopy())
    return 0;

  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto enc = cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  auto order = enc.getOrder();
  auto smeWpt = enc.getSmeWarpsPerCTA();
  if (smeWpt.size() != 2 || order.size() != 2)
    return 0;

  // Src elements are pointers; the loaded dtype lives on the memdesc result.
  auto dstTy = cast<ttg::MemDescType>(op.getResult().getType());
  Type elemTy = dstTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return 0;
  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  if (elemBytes == 0)
    return 0;

  auto shape = srcTy.getShape();

  bool isRowMajor = order[0] != 0;
  unsigned offset0 = isRowMajor ? 16 : 64 / elemBytes;
  unsigned offset1 = isRowMajor ? 64 / elemBytes : 16;
  if (smeWpt[0] == 0 || smeWpt[1] == 0 || offset0 == 0 || offset1 == 0)
    return 0;

  unsigned shapePerCTA0 = smeWpt[0] * offset0;
  unsigned shapePerCTA1 = smeWpt[1] * offset1;
  if (shapePerCTA0 == 0 || shapePerCTA1 == 0 || shape[0] % shapePerCTA0 != 0 ||
      shape[1] % shapePerCTA1 != 0)
    return 0;

  return static_cast<int>((shape[0] / shapePerCTA0) *
                          (shape[1] / shapePerCTA1));
}

int getOpNumberOfAsyncLoadInstructions(Operation *op) {
  if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
    return getIluvatarSmeG2SCost(copyOp);
  return 0;
}

using MemoCache = llvm::DenseSet<std::tuple<Operation *, int, int>>;
int computeMinCountBackward(Operation *cursor, Operation *cameFrom,
                            int numOutstanding, int pathSum, int bestPath,
                            MemoCache &branchStateCache,
                            llvm::function_ref<int(Operation *)> countFunc) {
  assert(cameFrom != nullptr);
  auto getPredecessor = [&cameFrom](Operation *op) {
    auto prevOp = op->getPrevNode();
    if (!prevOp) {
      prevOp = op->getParentOp();
      if (isa<ModuleOp>(prevOp)) {
        prevOp = nullptr;
      }
    }
    return prevOp;
  };

  auto continueWalkFrom = [&](Operation *newCursor) {
    auto pathResult =
        computeMinCountBackward(newCursor, cursor, numOutstanding, pathSum,
                                bestPath, branchStateCache, countFunc);
    bestPath = std::min(bestPath, pathResult);
    return pathResult;
  };

  while (cursor) {
    if (numOutstanding < 0 || pathSum >= bestPath) {
      return std::min(bestPath, pathSum);
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(cursor)) {
      bool cameFromThenOrElse = cameFrom->getParentOp() == ifOp;
      if (cameFromThenOrElse) {
        continueWalkFrom(getPredecessor(ifOp));
      } else {
        continueWalkFrom(ifOp.getThenRegion().front().getTerminator());
        if (!ifOp.getElseRegion().empty()) {
          continueWalkFrom(ifOp.getElseRegion().front().getTerminator());
        } else {
          continueWalkFrom(getPredecessor(ifOp));
        }
      }
      return bestPath;
    } else if (auto forOp = dyn_cast<scf::ForOp>(cursor)) {
      continueWalkFrom(getPredecessor(forOp));

      auto cameFromBody = cameFrom->getBlock() == forOp.getBody();
      auto cacheKey = std::make_tuple(cursor, numOutstanding, pathSum);
      if (!cameFromBody || branchStateCache.insert(cacheKey).second) {
        continueWalkFrom(forOp.getBody()->getTerminator());
      }
      return bestPath;
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(cursor)) {
      Block *lastBlock = cameFrom->getBlock();
      bool cameFromBefore = lastBlock == whileOp.getBeforeBody();
      bool cameFromAfter = lastBlock == whileOp.getAfterBody();
      bool cameFromSuccessor = !cameFromAfter && !cameFromBefore;

      if (cameFromAfter || cameFromSuccessor) {
        continueWalkFrom(whileOp.getBeforeBody()->getTerminator());
      } else if (cameFromBefore) {
        continueWalkFrom(getPredecessor(whileOp));
        auto cacheKey = std::make_tuple(cursor, numOutstanding, pathSum);
        if (branchStateCache.insert(cacheKey).second)
          continueWalkFrom(whileOp.getAfterBody()->getTerminator());
      }
      return bestPath;
    } else if (isa<triton::FuncOp>(cursor)) {
      return std::min(bestPath, pathSum);
    } else if (cursor->getNumRegions() > 0 && !isa<triton::ReduceOp>(cursor)) {
      cursor->emitRemark(
          "has subregions but is not analyzed when determining async "
          "wait count; this yields conservative waits");
      return 0;
    }

    pathSum += countFunc(cursor);
    if (isa<ttg::AsyncCommitGroupOp>(cursor)) {
      numOutstanding--;
    }

    cameFrom = cursor;
    cursor = getPredecessor(cursor);
  }
  return std::min(pathSum, bestPath);
}

int computeMinCountBackward(ttg::AsyncWaitOp waitOp,
                            llvm::function_ref<int(Operation *)> countFunc) {
  MemoCache memoCache;
  return computeMinCountBackward(waitOp, waitOp, waitOp.getNum(), 0,
                                 std::numeric_limits<int>::max(), memoCache,
                                 countFunc);
}

void updateWaitCount(ttg::AsyncWaitOp waitOp,
                     llvm::function_ref<int(Operation *)> computeCountForOp,
                     RewriterBase &rewriter) {
  int waitCnt = std::numeric_limits<int>::max();

  if (waitOp.getNumOperands() > 0) {
    for (auto token : waitOp.getOperands()) {
      auto tokenWaitCnt =
          deduceMinCountOnDefChain(token, waitOp, computeCountForOp);
      waitCnt = std::min(waitCnt, tokenWaitCnt);
    }
  } else {
    waitCnt = computeMinCountBackward(waitOp, computeCountForOp);
  }

  if (waitCnt == std::numeric_limits<int>::max()) {
    waitCnt = 0;
  }

  // Keep ttg.async_wait but reinterpret num as outstanding G2S intrinsics
  rewriter.modifyOpInPlace(waitOp, [&]() {
    waitOp.setNum(waitCnt);
    waitOp->setAttr(kIluvatarG2SWaitAttrName, rewriter.getUnitAttr());
  });
}

} // anonymous namespace

struct TritonILUVATARGPUUpdateAsyncWaitCountPass
    : impl::TritonILUVATARGPUUpdateAsyncWaitCountBase<
          TritonILUVATARGPUUpdateAsyncWaitCountPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<ttg::AsyncWaitOp> waitOps;
    getOperation()->walk([&](ttg::AsyncWaitOp waitOp) {
      if (!waitOp->hasAttr(kIluvatarG2SWaitAttrName))
        waitOps.push_back(waitOp);
    });

    DenseMap<Operation *, int> intrinsicCountCache;
    auto countAsyncLoadInstructions = [&](Operation *op) {
      auto found = intrinsicCountCache.find(op);
      if (found != intrinsicCountCache.end())
        return found->second;
      auto v = getOpNumberOfAsyncLoadInstructions(op);
      intrinsicCountCache[op] = v;
      return v;
    };

    for (auto waitOp : waitOps) {
      IRRewriter builder(waitOp->getContext());
      updateWaitCount(waitOp, countAsyncLoadInstructions, builder);
    }
  }
};

std::unique_ptr<Pass> createTritonILUVATARGPUUpdateAsyncWaitCountPass() {
  return std::make_unique<TritonILUVATARGPUUpdateAsyncWaitCountPass>();
}

} // namespace mlir
