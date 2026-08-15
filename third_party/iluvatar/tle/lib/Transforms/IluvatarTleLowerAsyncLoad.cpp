// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "Transforms/Passes.h"

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

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLELOWERASYNCLOAD
#include "Transforms/Passes.h.inc"

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

// A dynamic (runtime) splat mask cannot be represented by the async copy path,
// which only understands element-wise masks; bail out and keep the sync load.
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

// Async copy only applies to loads from a tensor of global-memory pointers.
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

static ttg::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder,
                                                 tt::LoadOp op) {
  auto resultTy = cast<RankedTensorType>(op.getType());
  auto sharedEncoding = tt::getSharedEncoding(op);
  auto sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(builder.getContext());
  auto memDescTy = ttg::MemDescType::get(
      resultTy.getShape(), resultTy.getElementType(), sharedEncoding,
      sharedMemorySpace, /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(builder, op.getLoc(), memDescTy);
}

// Rewrite a `tt.load` marked with `tt.load.async = true` into the async
// global-to-shared copy chain that the Iluvatar backend now lowers:
//   ttg.local_alloc
//   ttg.async_copy_global_to_local -> ttg.async_commit_group
//   ttg.async_wait -> ttg.local_load (at the first use).
static void lowerAsyncLoad(tt::LoadOp op, RewriterBase &rewriter,
                           tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  auto alloc = createLocalAllocForLoad(rewriter, op);
  auto copy = ttg::AsyncCopyGlobalToLocalOp::create(
      rewriter, loc, op.getPtr(), alloc.getResult(), op.getMask(),
      op.getOther(), op.getInputStride(), op.getCache(), op.getEvict(),
      op.getIsVolatile(), getAsyncLoadContiguity(op, axisInfoAnalysis));
  auto commit = ttg::AsyncCommitGroupOp::create(rewriter, loc, copy.getToken());

  Operation *firstUse = getFirstUseInSameBlock(op);
  assert(firstUse && "async load should have a same-block use");
  rewriter.setInsertionPoint(firstUse);
  auto wait = ttg::AsyncWaitOp::create(rewriter, loc, commit.getResult(), 0);

  if (hasNonZeroOther(op) && op.getMask()) {
    // AsyncCopyGlobalToLocalOp does not preserve a non-zero `other`, so restore
    // it with an explicit select over the shared load result.
    auto localLoad = ttg::LocalLoadOp::create(
        rewriter, loc, op.getType(), alloc.getResult(), wait.getResult());
    auto select =
        arith::SelectOp::create(rewriter, loc, op.getType(), op.getMask(),
                                localLoad.getResult(), op.getOther());
    op.getResult().replaceAllUsesWith(select.getResult());
  } else {
    tt::replaceUsesWithLocalLoad(rewriter, op->getResult(0), alloc.getResult(),
                                 wait.getResult());
  }

  rewriter.eraseOp(op);
}

} // namespace

class LowerAsyncLoadPass
    : public impl::TritonIluvatarTleLowerAsyncLoadBase<LowerAsyncLoadPass> {
public:
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
      // `tle.load(..., is_async=True)` is only a scheduling hint: whenever the
      // async copy path is not applicable we simply drop the attribute and keep
      // the ordinary synchronous `tt.load` semantics.
      if (!isTrueAsyncLoadAttr(op) ||
          !canLowerAsyncLoad(op, axisInfoAnalysis)) {
        op->removeAttr(kAsyncLoadAttr);
        continue;
      }
      lowerAsyncLoad(op, rewriter, axisInfoAnalysis);
    }
  }
};

} // namespace mlir::triton::iluvatar_tle
