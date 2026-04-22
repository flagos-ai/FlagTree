#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>

namespace mlir {
namespace triton {
namespace tle {

#define GEN_PASS_DEF_TRITONTLEDOWNGRADEINVALIDASYNCCOPY
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static unsigned product(ArrayRef<unsigned> values) {
  unsigned result = 1;
  for (unsigned value : values)
    result *= value;
  return result;
}

static bool hasLegalCpAsyncWidth(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                 tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto dstTy = cast<ttg::MemDescType>(copyOp.getResult().getType());
  Type elemTy = dstTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return true;

  unsigned elemBits = elemTy.getIntOrFloatBitWidth();
  if (elemBits == 0)
    return false;

  unsigned maxVec = axisInfo.getContiguity(copyOp.getSrc());
  if (Value mask = copyOp.getMask())
    maxVec = std::min<unsigned>(maxVec, axisInfo.getMaskAlignment(mask));
  maxVec = std::max<unsigned>(maxVec, copyOp.getContiguity());
  maxVec = std::min<unsigned>(maxVec, 128 / elemBits);

  unsigned vecBytes = maxVec * elemBits / 8;
  return llvm::is_contained({4u, 8u, 16u}, vecBytes);
}

static unsigned floorPowerOfTwo(unsigned value) {
  if (value == 0)
    return 0;
  return 1u << llvm::Log2_32(value);
}

static unsigned getLegalCpAsyncVec(unsigned maxVec, unsigned elemBits) {
  if (elemBits == 0)
    return 0;

  maxVec = std::min<unsigned>(maxVec, 128 / elemBits);
  maxVec = floorPowerOfTwo(maxVec);
  while (maxVec > 0) {
    unsigned vecBytes = maxVec * elemBits / 8;
    if (llvm::is_contained({4u, 8u, 16u}, vecBytes))
      return maxVec;
    maxVec >>= 1;
  }
  return 0;
}

static unsigned
getMaxVectorElementsOnAxis(Value value, unsigned axis, unsigned elemBits,
                           tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return 1;
  auto *info = axisInfo.getAxisInfo(value);
  if (!info || axis >= static_cast<unsigned>(info->getRank()))
    return 1;

  auto elemBytes = std::max<unsigned>(elemBits / 8, 1);
  unsigned contiguity = std::max<int64_t>(info->getContiguity(axis), 1);
  unsigned divisibility = std::max<int64_t>(info->getDivisibility(axis), 1);
  unsigned maxMultiple = divisibility;
  if (isa<tt::PointerType>(tensorTy.getElementType()))
    maxMultiple = std::max<unsigned>(divisibility / elemBytes, 1);
  return std::min(contiguity, maxMultiple);
}

static std::optional<ttg::BlockedEncodingAttr>
getCoalescedFallbackLoadEncoding(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                 tt::ModuleAxisInfoAnalysis &axisInfo) {
  if (!copyOp->hasAttr(kTleLocalPointerAsyncStoreAttr))
    return std::nullopt;

  auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto dstTy = dyn_cast<ttg::MemDescType>(copyOp.getResult().getType());
  if (!srcTy || !dstTy)
    return std::nullopt;
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  if (!blockedEnc)
    return std::nullopt;

  Type elemTy = dstTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned elemBits = elemTy.getIntOrFloatBitWidth();
  if (elemBits == 0)
    return std::nullopt;

  auto *ptrInfo = axisInfo.getAxisInfo(copyOp.getSrc());
  if (!ptrInfo)
    return std::nullopt;

  SmallVector<unsigned> order(srcTy.getRank());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return ptrInfo->getContiguity(lhs) > ptrInfo->getContiguity(rhs);
  });

  Value mask = copyOp.getMask();
  auto *maskInfo = mask ? axisInfo.getAxisInfo(mask) : nullptr;
  auto sizePerThread = llvm::to_vector(blockedEnc.getSizePerThread());

  for (unsigned axis : order) {
    unsigned maxVec =
        getMaxVectorElementsOnAxis(copyOp.getSrc(), axis, elemBits, axisInfo);
    if (maskInfo && axis < static_cast<unsigned>(maskInfo->getRank()))
      maxVec = std::min<unsigned>(
          maxVec, std::max<int64_t>(maskInfo->getConstancy(axis), 1));
    maxVec = std::min<unsigned>(maxVec, srcTy.getShape()[axis]);

    unsigned legalVec = getLegalCpAsyncVec(maxVec, elemBits);
    if (legalVec == 0 || legalVec <= sizePerThread[axis])
      continue;

    SmallVector<unsigned> newSizePerThread(sizePerThread);
    newSizePerThread[axis] = legalVec;
    unsigned numWarps = product(blockedEnc.getWarpsPerCTA());
    unsigned threadsPerWarp = product(blockedEnc.getThreadsPerWarp());
    auto newEnc = ttg::BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), newSizePerThread, order,
        numWarps, threadsPerWarp, blockedEnc.getCTALayout());
    return newEnc;
  }

  return std::nullopt;
}

static Value convertTensorEncoding(IRRewriter &rewriter, Location loc,
                                   Value value, Attribute encoding) {
  if (!value)
    return value;
  auto ty = dyn_cast<RankedTensorType>(value.getType());
  if (!ty || ty.getEncoding() == encoding)
    return value;
  auto newTy = ty.cloneWithEncoding(encoding);
  return ttg::ConvertLayoutOp::create(rewriter, loc, newTy, value).getResult();
}

static LogicalResult
collectErasableTokenUsers(Value token,
                          SmallPtrSetImpl<Operation *> &visitedTokenOps,
                          SmallPtrSetImpl<Operation *> &localLoadSet,
                          SmallVectorImpl<ttg::LocalLoadOp> &localLoads,
                          SmallVectorImpl<Operation *> &eraseOps) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : token.getUses())
    uses.push_back(&use);

  for (OpOperand *use : uses) {
    Operation *user = use->getOwner();
    if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(user)) {
      if (localLoadSet.insert(user).second)
        localLoads.push_back(localLoad);
      continue;
    }

    if (!isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(user))
      return failure();

    // Rewriting a token out of a multi-token commit/wait would require
    // rebuilding the remaining async dependency chain. Keep this pass
    // conservative and only erase self-contained single-token chains.
    if (user->getNumOperands() != 1)
      return failure();

    if (visitedTokenOps.insert(user).second) {
      if (failed(collectErasableTokenUsers(user->getResult(0), visitedTokenOps,
                                           localLoadSet, localLoads, eraseOps)))
        return failure();
      eraseOps.push_back(user);
    }
  }

  return success();
}

static void recreateLocalLoadWithoutToken(IRRewriter &rewriter,
                                          ttg::LocalLoadOp localLoad) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(localLoad);
  auto replacement = ttg::LocalLoadOp::create(
      rewriter, localLoad.getLoc(), localLoad.getType(), localLoad.getSrc());
  replacement->setAttrs(localLoad->getAttrs());
  localLoad.replaceAllUsesWith(replacement.getResult());
  rewriter.eraseOp(localLoad);
}

static LogicalResult downgradeAsyncCopy(IRRewriter &rewriter,
                                        ttg::AsyncCopyGlobalToLocalOp copyOp,
                                        tt::ModuleAxisInfoAnalysis &axisInfo) {
  SmallPtrSet<Operation *, 8> visitedTokenOps;
  SmallPtrSet<Operation *, 8> localLoadSet;
  SmallVector<ttg::LocalLoadOp> localLoads;
  SmallVector<Operation *> eraseOps;
  if (failed(collectErasableTokenUsers(copyOp.getToken(), visitedTokenOps,
                                       localLoadSet, localLoads, eraseOps)))
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(copyOp);
  auto originalSrcTy = cast<RankedTensorType>(copyOp.getSrc().getType());
  Value src = copyOp.getSrc();
  Value mask = copyOp.getMask();
  Value other = copyOp.getOther();
  if (auto encoding = getCoalescedFallbackLoadEncoding(copyOp, axisInfo)) {
    src = convertTensorEncoding(rewriter, copyOp.getLoc(), src, *encoding);
    mask = convertTensorEncoding(rewriter, copyOp.getLoc(), mask, *encoding);
    other = convertTensorEncoding(rewriter, copyOp.getLoc(), other, *encoding);
  }
  auto load = tt::LoadOp::create(
      rewriter, copyOp.getLoc(), src, mask, other, copyOp.getCache(),
      copyOp.getEvict(), copyOp.getIsVolatile(), rewriter.getStringAttr(""));
  Value storeValue = load.getResult();
  auto loadTy = cast<RankedTensorType>(storeValue.getType());
  if (loadTy.getEncoding() != originalSrcTy.getEncoding()) {
    auto storeFriendlyTy =
        loadTy.cloneWithEncoding(originalSrcTy.getEncoding());
    storeValue = ttg::ConvertLayoutOp::create(rewriter, copyOp.getLoc(),
                                              storeFriendlyTy, storeValue)
                     .getResult();
  }
  ttg::LocalStoreOp::create(rewriter, copyOp.getLoc(), storeValue,
                            copyOp.getResult());

  for (ttg::LocalLoadOp localLoad : localLoads)
    recreateLocalLoadWithoutToken(rewriter, localLoad);
  for (Operation *op : eraseOps)
    rewriter.eraseOp(op);
  rewriter.eraseOp(copyOp);

  return success();
}

struct DowngradeInvalidAsyncCopyPass
    : impl::TritonTleDowngradeInvalidAsyncCopyBase<
          DowngradeInvalidAsyncCopyPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    tt::ModuleAxisInfoAnalysis axisInfo(module);

    SmallVector<ttg::AsyncCopyGlobalToLocalOp> invalidCopies;
    module.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
      if (hasLegalCpAsyncWidth(copyOp, axisInfo))
        return;
      invalidCopies.push_back(copyOp);
    });

    IRRewriter rewriter(&getContext());
    for (ttg::AsyncCopyGlobalToLocalOp copyOp : invalidCopies) {
      if (copyOp->getBlock() &&
          failed(downgradeAsyncCopy(rewriter, copyOp, axisInfo))) {
        copyOp.emitError("cannot downgrade invalid async copy with non-trivial "
                         "async token users");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace tle
} // namespace triton
} // namespace mlir
