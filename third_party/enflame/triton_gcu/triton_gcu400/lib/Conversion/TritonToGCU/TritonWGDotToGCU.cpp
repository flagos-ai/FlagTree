/**
 * Copyright 2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONWGDOTTOGCU
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
struct TritonWGDotToGCUPass
    : public mlir::impl::TritonWGDotToGCUBase<TritonWGDotToGCUPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

static Attribute convertResultEncodingToBlocked(RankedTensorType tensorType) {
  auto enc = tensorType.getEncoding();
  if (auto blockedEnc = dyn_cast<BlockedEncodingAttr>(enc))
    return blockedEnc;

  if (auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(enc)) {
    auto ctx = tensorType.getContext();
    unsigned rank = tensorType.getRank();
    SmallVector<unsigned> sizePerThread(rank, 1);
    SmallVector<unsigned> threadsPerWarp(rank, 1);
    threadsPerWarp[rank - 1] = 32;
    SmallVector<unsigned> order(rank);
    for (unsigned i = 0; i < rank; ++i)
      order[i] = rank - 1 - i;
    auto ctaLayout = getCTALayout(enc);
    return BlockedEncodingAttr::get(ctx, sizePerThread, threadsPerWarp,
                                    mmaEnc.getWarpsPerCTA(), order, ctaLayout);
  }

  return nullptr;
}

static Attribute getBlockedEncoding(triton::gcu::WarpGroupDotOp wgDotOp) {
  auto retType = cast<RankedTensorType>(wgDotOp.getD().getType());
  if (auto blockedEnc = convertResultEncodingToBlocked(retType))
    return blockedEnc;

  auto accType = cast<RankedTensorType>(wgDotOp.getC().getType());
  if (auto blockedEnc = convertResultEncodingToBlocked(accType))
    return blockedEnc;

  return nullptr;
}

class WarpGroupDotToDotPattern
    : public OpRewritePattern<triton::gcu::WarpGroupDotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::WarpGroupDotOp wgDotOp,
                                PatternRewriter &rewriter) const override {
    auto loc = wgDotOp.getLoc();
    auto ctx = wgDotOp.getContext();

    auto blockedEnc = getBlockedEncoding(wgDotOp);
    if (!blockedEnc)
      return failure();

    auto oldRetType = wgDotOp.getD().getType();
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), blockedEnc);

    auto convertOperand = [&](Value operand, int opIdx) -> Value {
      // If the operand is already a tensor with DotOperandEncodingAttr,
      // no conversion needed — it was set up by AccelerateMatmul.
      if (auto tensorTy = dyn_cast<RankedTensorType>(operand.getType())) {
        if (isa<DotOperandEncodingAttr>(tensorTy.getEncoding()))
          return operand;
        auto dotOpEnc = DotOperandEncodingAttr::get(ctx, opIdx, blockedEnc,
                                                    tensorTy.getElementType());
        auto newTy = RankedTensorType::get(tensorTy.getShape(),
                                           tensorTy.getElementType(), dotOpEnc);
        if (tensorTy == newTy)
          return operand;
        return rewriter.create<ConvertLayoutOp>(loc, newTy, operand);
      }
      // SharedMemory MemDesc — emit LocalLoadOp.
      auto memDescTy = cast<MemDescType>(operand.getType());
      auto dotOpEnc = DotOperandEncodingAttr::get(ctx, opIdx, blockedEnc,
                                                  memDescTy.getElementType());
      auto loadTy = RankedTensorType::get(memDescTy.getShape(),
                                          memDescTy.getElementType(), dotOpEnc);
      return rewriter.create<triton::gpu::LocalLoadOp>(loc, loadTy, operand);
    };

    Value newA = convertOperand(wgDotOp.getA(), 0);
    Value newB = convertOperand(wgDotOp.getB(), 1);

    // Look through ConvertLayoutOp on the accumulator to avoid
    // redundant round-trips (e.g. #blocked -> #mma -> #blocked1).
    Value oldAcc = wgDotOp.getC();
    if (auto cvtOp = oldAcc.getDefiningOp<ConvertLayoutOp>()) {
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      if (srcType.getShape() == oldRetType.getShape() &&
          srcType.getElementType() == oldRetType.getElementType())
        oldAcc = cvtOp.getSrc();
    }

    Value newAcc = oldAcc;
    auto accType = cast<RankedTensorType>(oldAcc.getType());
    if (accType.getEncoding() != blockedEnc) {
      auto accTargetType = RankedTensorType::get(
          accType.getShape(), accType.getElementType(), blockedEnc);
      newAcc = rewriter.create<ConvertLayoutOp>(loc, accTargetType, oldAcc);
    }

    auto newDot = rewriter.create<DotOp>(loc, newRetType, newA, newB, newAcc,
                                         wgDotOp.getInputPrecision(),
                                         wgDotOp.getMaxNumImpreciseAcc());

    Value dotResult = newDot.getResult();

    if (newRetType == oldRetType) {
      rewriter.replaceOp(wgDotOp, dotResult);
    } else {
      // Fold downstream ConvertLayoutOp users: convert directly from
      // dotResult (#blocked1) to their target type, skipping the
      // intermediate old layout.
      Value oldTypeResult = nullptr;
      for (OpOperand &use :
           llvm::make_early_inc_range(wgDotOp.getD().getUses())) {
        if (auto cvtUser = dyn_cast<ConvertLayoutOp>(use.getOwner())) {
          auto targetType = cvtUser.getType();
          if (targetType == newRetType) {
            rewriter.replaceOp(cvtUser, dotResult);
          } else {
            rewriter.replaceOpWithNewOp<ConvertLayoutOp>(cvtUser, targetType,
                                                         dotResult);
          }
          continue;
        }
        if (!oldTypeResult)
          oldTypeResult =
              rewriter.create<ConvertLayoutOp>(loc, oldRetType, dotResult);
        use.set(oldTypeResult);
      }
      rewriter.eraseOp(wgDotOp);
    }

    return success();
  }
};

// Match triton_gcu.load followed by ttg.local_alloc and insert a
// triton_gcu.load_global_to_share that fuses them into a DMA transfer.
class LoadAllocToLoadGlobalToSharePattern
    : public OpRewritePattern<triton::gcu::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.getResult().hasOneUse())
      return failure();
    auto localAllocOp = dyn_cast<triton::gpu::LocalAllocOp>(
        *loadOp.getResult().getUsers().begin());
    if (!localAllocOp || localAllocOp.getSrc() != loadOp.getResult())
      return failure();

    if (auto afterAlloc = localAllocOp->getNextNode())
      if (auto copyOp = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(afterAlloc))
        if (copyOp.getDstMem() == localAllocOp.getResult())
          return failure();

    rewriter.setInsertionPointAfter(localAllocOp);
    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        loadOp.getLoc(), loadOp.getPtr(), loadOp.getShape(),
        loadOp.getStrides(), loadOp.getOffsets(), localAllocOp.getResult(),
        loadOp.getDefaultValue(), loadOp.getOrderHint());
    rewriter.modifyOpInPlace(localAllocOp, [&]() {
      localAllocOp->setOperands(ValueRange{});
      auto oldType =
          cast<triton::gpu::MemDescType>(localAllocOp.getResult().getType());
      auto mutableType = triton::gpu::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), /*mutableMemory=*/true);
      localAllocOp.getResult().setType(mutableType);
    });
    rewriter.eraseOp(loadOp);

    return success();
  }
};

// Match triton_gcu.load followed by ttg.local_store and insert a
// triton_gcu.load_global_to_share that fuses them into a DMA transfer.
class LoadStoreToLoadGlobalToSharePattern
    : public OpRewritePattern<triton::gcu::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.getResult().hasOneUse())
      return failure();
    auto localStoreOp = dyn_cast<triton::gpu::LocalStoreOp>(
        *loadOp.getResult().getUsers().begin());
    if (!localStoreOp || localStoreOp.getSrc() != loadOp.getResult())
      return failure();

    rewriter.setInsertionPointAfter(loadOp);
    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        loadOp.getLoc(), loadOp.getPtr(), loadOp.getShape(),
        loadOp.getStrides(), loadOp.getOffsets(), localStoreOp.getDst(),
        loadOp.getDefaultValue(), loadOp.getOrderHint());
    rewriter.eraseOp(localStoreOp);
    rewriter.eraseOp(loadOp);

    return success();
  }
};

void TritonWGDotToGCUPass::runOnOperation() {
  auto *ctx = &getContext();
  mlir::gpu::GPUModuleOp m = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<LoadAllocToLoadGlobalToSharePattern,
               LoadStoreToLoadGlobalToSharePattern, WarpGroupDotToDotPattern>(
      ctx);
  if (applyPatternsGreedily(m, std::move(patterns)).failed())
    signalPassFailure();
}
