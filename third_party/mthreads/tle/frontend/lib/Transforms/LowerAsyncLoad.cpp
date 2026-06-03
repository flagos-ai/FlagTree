#include "MUSATLE/Frontend/Passes.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kAsyncLoadAttr = "tt.load.async";

static bool isTrueAsyncLoadAttr(tt::LoadOp op) {
  auto attr = op->getAttrOfType<BoolAttr>(kAsyncLoadAttr);
  return attr && attr.getValue();
}

static bool isSplatConstantTrue(Value value) {
  auto splat = value.getDefiningOp<tt::SplatOp>();
  if (!splat)
    return false;
  return isConstantIntValue(splat.getSrc(), 1);
}

static bool hasDynamicSplatMask(tt::LoadOp op) {
  Value mask = op.getMask();
  if (!mask)
    return false;
  auto splat = mask.getDefiningOp<tt::SplatOp>();
  return splat && !isSplatConstantTrue(mask);
}

static bool hasSupportedElementType(RankedTensorType type) {
  Type elemTy = type.getElementType();
  if (!elemTy.isIntOrFloat())
    return false;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  return bitWidth == 8 || bitWidth == 16 || bitWidth == 32 || bitWidth == 64;
}

static bool hasTensorPointerSource(tt::LoadOp op) {
  if (isLoadFromTensorPtr(op))
    return false;

  auto ptrTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
  if (!ptrTy)
    return false;

  auto elemPtrTy = dyn_cast<tt::PointerType>(ptrTy.getElementType());
  return elemPtrTy && elemPtrTy.getAddressSpace() == 1;
}

static Operation *getFirstUseInSameBlock(tt::LoadOp op) {
  Operation *firstUse = nullptr;
  Block *block = op->getBlock();
  for (Operation *user : op->getUsers()) {
    if (user->getBlock() != block)
      return nullptr;
    if (!firstUse || user->isBeforeInBlock(firstUse))
      firstUse = user;
  }
  return firstUse;
}

static unsigned
getAsyncLoadContiguity(tt::LoadOp op,
                       tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  Value ptr = op.getPtr();
  unsigned contiguity = axisInfoAnalysis.getContiguity(ptr);
  if (Value mask = op.getMask())
    contiguity =
        std::min<unsigned>(contiguity, axisInfoAnalysis.getMaskAlignment(mask));
  return std::max(1u, contiguity);
}

static bool canLowerAsyncLoad(tt::LoadOp op,
                              tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  if (op->use_empty())
    return false;
  auto resultTy = dyn_cast<RankedTensorType>(op.getType());
  if (!resultTy || !hasSupportedElementType(resultTy))
    return false;
  if (!hasTensorPointerSource(op))
    return false;
  if (op.getIsVolatile())
    return false;
  if (hasDynamicSplatMask(op))
    return false;
  if (!getFirstUseInSameBlock(op))
    return false;

  return tt::canBeAsyncLoad(op) &&
         tt::canBeConvertedToAsyncLoad(op, axisInfoAnalysis);
}

static bool hasNonZeroOther(tt::LoadOp op) {
  return op.getOther() && !isZeroConst(op.getOther());
}

static bool getSqmmaAttrValue(Operation *op, llvm::StringRef name,
                              Attribute &value) {
  value = op->getAttr(name);
  return static_cast<bool>(value);
}

static bool haveSameSqmmaAttrs(Operation *lhs, Operation *rhs) {
  for (auto name : tt::musa::kSqmmaAttrNames) {
    Attribute lhsAttr;
    Attribute rhsAttr;
    bool lhsHas = getSqmmaAttrValue(lhs, name, lhsAttr);
    bool rhsHas = getSqmmaAttrValue(rhs, name, rhsAttr);
    if (lhsHas != rhsHas)
      return false;
    if (lhsHas && lhsAttr != rhsAttr)
      return false;
  }
  return true;
}

struct ForwardedLocalAllocInfo {
  bool canForward = true;
  std::optional<ttg::LocalAllocOp> attrSource;
};

static ForwardedLocalAllocInfo
getForwardedLocalAllocInfo(tt::LoadOp op, ttg::MemDescType dstTy) {
  ForwardedLocalAllocInfo info;
  std::optional<ttg::LocalAllocOp> attrSource;

  for (Operation *user : op->getUsers()) {
    auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
    if (!localAlloc)
      continue;

    auto userTy = localAlloc.getType();
    if (userTy.getEncoding() != dstTy.getEncoding())
      continue;

    if (!attrSource) {
      attrSource = localAlloc;
      continue;
    }
    if (!haveSameSqmmaAttrs(attrSource->getOperation(),
                            localAlloc.getOperation())) {
      info.canForward = false;
      return info;
    }
  }

  info.attrSource = attrSource;
  return info;
}

static ttg::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder,
                                                 tt::LoadOp op,
                                                 bool &canForwardLocalAllocs) {
  auto resultTy = cast<RankedTensorType>(op.getType());
  auto sharedEncoding = tt::getSharedEncoding(op);
  auto sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(builder.getContext());
  auto memDescTy = ttg::MemDescType::get(
      resultTy.getShape(), resultTy.getElementType(), sharedEncoding,
      sharedMemorySpace, /*mutableMemory=*/true);

  auto alloc = ttg::LocalAllocOp::create(builder, op.getLoc(), memDescTy);
  auto forwardInfo = getForwardedLocalAllocInfo(op, memDescTy);
  canForwardLocalAllocs = forwardInfo.canForward;
  if (forwardInfo.attrSource)
    tt::musa::copySqmmaAttrs(forwardInfo.attrSource->getOperation(),
                             alloc.getOperation());
  return alloc;
}

static void lowerAsyncLoad(tt::LoadOp op, RewriterBase &rewriter,
                           tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  bool canForwardLocalAllocs = true;
  auto alloc = createLocalAllocForLoad(rewriter, op, canForwardLocalAllocs);
  auto copy = ttg::AsyncCopyGlobalToLocalOp::create(
      rewriter, loc, op.getPtr(), alloc.getResult(), op.getMask(),
      op.getOther(), op.getCache(), op.getEvict(), op.getIsVolatile(),
      getAsyncLoadContiguity(op, axisInfoAnalysis));
  auto commit = ttg::AsyncCommitGroupOp::create(rewriter, loc, copy.getToken());
  Operation *firstUse = getFirstUseInSameBlock(op);
  assert(firstUse && "async load should have a same-block use");
  rewriter.setInsertionPoint(firstUse);
  auto wait = ttg::AsyncWaitOp::create(rewriter, loc, commit.getResult(), 0);

  if (hasNonZeroOther(op) && op.getMask()) {
    auto localLoad = ttg::LocalLoadOp::create(
        rewriter, loc, op.getType(), alloc.getResult(), wait.getResult());
    auto select =
        arith::SelectOp::create(rewriter, loc, op.getType(), op.getMask(),
                                localLoad.getResult(), op.getOther());
    op.getResult().replaceAllUsesWith(select.getResult());
  } else if (canForwardLocalAllocs) {
    tt::replaceUsesWithLocalLoad(rewriter, op->getResult(0), alloc.getResult(),
                                 wait.getResult());
  } else {
    auto localLoad = ttg::LocalLoadOp::create(
        rewriter, loc, op.getType(), alloc.getResult(), wait.getResult());
    op.getResult().replaceAllUsesWith(localLoad.getResult());
  }

  rewriter.eraseOp(op);
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERASYNCLOAD
#include "MUSATLE/Frontend/Passes.h.inc"

struct TritonMUSAGPUTLELowerAsyncLoadPass
    : impl::TritonMUSAGPUTLELowerAsyncLoadBase<
          TritonMUSAGPUTLELowerAsyncLoadPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    SmallVector<tt::LoadOp> loadOps;
    mod.walk([&](tt::LoadOp op) {
      if (op->hasAttr(kAsyncLoadAttr))
        loadOps.push_back(op);
    });

    for (tt::LoadOp op : loadOps) {
      if (!isTrueAsyncLoadAttr(op)) {
        op->removeAttr(kAsyncLoadAttr);
        continue;
      }

      if (!canLowerAsyncLoad(op, axisInfoAnalysis)) {
        op->removeAttr(kAsyncLoadAttr);
        continue;
      }

      lowerAsyncLoad(op, rewriter, axisInfoAnalysis);
    }
  }
};

} // namespace mlir
