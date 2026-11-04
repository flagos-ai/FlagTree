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
#include "Utility.h"
#include "llvm/Support/Casting.h"
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
#include "Utils/TritonVersionCompat.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONWGDOTTOGCU
#include "Conversion/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
struct TritonWGDotToGCUPass
    : public mlir::impl::TritonWGDotToGCUBase<TritonWGDotToGCUPass> {
  using Base::Base;

  void runOnOperation() override;
};
}  // namespace

// ConvertLayoutOp changes the data distribution across threads/warps iff the
// threads-per-warp decomposition differs between its source and its result.
static bool changesThreadsPerWarp(triton::gpu::ConvertLayoutOp cvtOp) {
  auto srcType = dyn_cast<RankedTensorType>(cvtOp.getSrc().getType());
  auto dstType = dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!srcType || !dstType)
    return false;
  return triton::gpu::getThreadsPerWarp(srcType) !=
         triton::gpu::getThreadsPerWarp(dstType);
}

static Attribute
convertResultEncodingToBlocked(RankedTensorType tensorType) {
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
    auto ctaLayout = triton_gcu::compat::getCGALayout(enc);
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

    auto convertOperand = [&](Value operand, int opIdx,
                              bool columnMajor) -> Value {
      // If the operand is already a tensor with DotOperandEncodingAttr
      // and columnMajor matches, no conversion needed.
      if (auto tensorTy = dyn_cast<RankedTensorType>(operand.getType())) {
        if (isa<DotOperandEncodingAttr>(tensorTy.getEncoding()))
          return operand;
        auto dotOpEnc = DotOperandEncodingAttr::get(ctx, opIdx, blockedEnc,
                                                    /*kWidth=*/0, columnMajor);
        auto newTy = RankedTensorType::get(tensorTy.getShape(),
                                           tensorTy.getElementType(), dotOpEnc);
        if (tensorTy == newTy)
          return operand;
        return rewriter.create<ConvertLayoutOp>(loc, newTy, operand);
      }
      // SharedMemory MemDesc — emit LocalLoadOp.
      auto memDescTy = cast<MemDescType>(operand.getType());
      auto dotOpEnc = DotOperandEncodingAttr::get(ctx, opIdx, blockedEnc,
                                                  /*kWidth=*/0, columnMajor);
      auto loadTy = RankedTensorType::get(memDescTy.getShape(),
                                          memDescTy.getElementType(), dotOpEnc);
      return rewriter.create<triton::gpu::LocalLoadOp>(loc, loadTy, operand);
    };

    bool rhsColMajor = wgDotOp->hasAttr(kRhsColumnMajor);
    Value newA = convertOperand(wgDotOp.getA(), 0, false);
    Value newB = convertOperand(wgDotOp.getB(), 1, rhsColMajor);

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
    if (wgDotOp->hasAttr(kRhsColumnMajor))
      newDot->setAttr(kRhsColumnMajor, wgDotOp->getAttr(kRhsColumnMajor));
    if (!wgDotOp->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
      Operation *insertPoint = nullptr;
      bool foundBarrier = false;
      Block *wgDotBlock = wgDotOp->getBlock();

      // Walk the result's user chain (SSA def-use, across scf yield /
      // enclosing region results). The sync must be placed before the first
      // consumer that redistributes the data across threads, i.e. a
      // convert_layout whose src and dst use a different threads-per-warp
      // decomposition. A barrier already on the path means the sync exists,
      // so stop and insert nothing.
      SmallVector<Value> worklist;
      SmallPtrSet<Value, 8> visited;
      worklist.push_back(wgDotOp.getD());
      while (!worklist.empty() && !insertPoint) {
        Value cur = worklist.pop_back_val();
        if (!visited.insert(cur).second)
          continue;
        for (Operation *user : cur.getUsers()) {
          if (auto cvtUser = dyn_cast<triton::gpu::ConvertLayoutOp>(user)) {
            if (changesThreadsPerWarp(cvtUser)) {
              Operation *curOp = user;
              while (curOp) {
                Block *curBlock = curOp->getBlock();
                if (curBlock == wgDotBlock) {
                  insertPoint = curOp;
                  break;
                } else {
                  curOp = curBlock->getParentOp();
                }
              }
              assert(insertPoint && "cannot find insertPoint for DotOp");
            }
            // Same-thread redistribution: look through this convert.
            worklist.push_back(cvtUser.getResult());
            continue;
          } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
            // The yielded value becomes the enclosing region op's result:
            // keep tracing from there.
            Operation *parent = yieldOp->getParentOp();
            if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent)) {
              unsigned operandIdx = 0;
              for (OpOperand &operand : yieldOp->getOpOperands()) {
                if (operand.get() == cur)
                  break;
                ++operandIdx;
              }
              assert(operandIdx < yieldOp->getNumOperands());
              worklist.push_back(parent->getResult(operandIdx));
            } else {
              insertPoint = yieldOp;
            }
            break;
          }
          // Follow the result of any other intermediate op on the data path.
          for (Value res : user->getResults())
            if (res.getType() == cur.getType())
              worklist.push_back(res);
        }
      }

      if (insertPoint) {
        // If a barrier already sits between the warp-group dot and the
        // intended sync point in the same block, the sync exists -- do not
        // insert another one.
        for (auto it = std::next(wgDotOp->getIterator()),
                  end = wgDotBlock->end();
             it != end && &*it != insertPoint; ++it) {
          if (isa<mlir::gpu::BarrierOp>(&*it)) {
            foundBarrier = true;
            break;
          }
        }

        if (!foundBarrier) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(insertPoint);
          rewriter.create<mlir::gpu::BarrierOp>(loc);
        }
      }
    }

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
            rewriter.replaceOpWithNewOp<ConvertLayoutOp>(
                cvtUser, targetType, dotResult);
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

void TritonWGDotToGCUPass::runOnOperation() {
  auto *ctx = &getContext();
  mlir::gpu::GPUModuleOp m = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<WarpGroupDotToDotPattern>(ctx);
  if (applyPatternsGreedily(m, std::move(patterns)).failed())
    signalPassFailure();
}
