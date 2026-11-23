/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "Utils/Utils.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "ascend/include/TritonToLinalg/LoadStoreConverter.h"
#include "ascend/include/TritonToLinalg/MaskAnalysis.h"
#include "ascend/include/TritonToStructured/MemOpConverter.h"
#include "ascend/include/TritonToUnstructure/OffsetAnalysis.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "discrete-mask-access-conversion"

using namespace mlir;
using namespace hivm;

// File-scope flags set by DiscreteMaskAccessConversionPass::runOnOperation()
// before pattern application, so that OpRewritePattern subclasses can read
// them.
static bool compileOn91095Flag = false;
static triton::ascend::CompileMode compileModeFlag =
    triton::ascend::CompileMode::Simd;
static bool useSyncBlockLockFlag = true;

// Only signed comparisons against the same fixed, zero-based range are
// accepted. Preserve the original bound's integer semantics, including
// negative values; MaskState clamps the eventual access extent.
static OpFoldResult getPrefixBound(Value mask, Value range) {
  auto cmp = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
      cmp.getLhs() != range)
    return {};
  if (auto splat = cmp.getRhs().getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  if (auto constant = cmp.getRhs().getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
    if (dense && dense.isSplat())
      return IntegerAttr::get(dense.getElementType(),
                              dense.getSplatValue<APInt>());
  }
  return {};
}

// (lane < bound) & (lane < limit) == lane < min(bound, limit).
// Carry the exact scalar boundary instead of a tensor<i1>. This retains the
// history even when limits grow again, and lets the existing range lowering
// keep using subview/copy. Do not change CFO-owned descriptor signatures.
struct CanonicalizeLoopPrefixMask : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop->hasAttr(triton::controlflow::kPointerDescriptorBoundaryAttr))
      return failure();

    std::optional<unsigned> maskSlot;
    for (auto [slot, init] : llvm::enumerate(loop.getInitArgs())) {
      auto type = dyn_cast<RankedTensorType>(init.getType());
      if (!type || !type.getElementType().isInteger(1))
        continue;
      if (maskSlot || type.getRank() != 1 || !type.hasStaticShape())
        return failure();
      maskSlot = slot;
    }
    if (!maskSlot)
      return failure();

    unsigned slot = *maskSlot;
    auto iterMask = loop.getRegionIterArgs()[slot];
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    auto update = yield.getOperand(slot).getDefiningOp<arith::AndIOp>();
    if (!update || update->getBlock() != loop.getBody())
      return failure();
    Value condition;
    if (update.getLhs() == iterMask)
      condition = update.getRhs();
    else if (update.getRhs() == iterMask)
      condition = update.getLhs();
    else
      return failure();

    auto cmp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return failure();
    auto range = cmp.getLhs().getDefiningOp<triton::MakeRangeOp>();
    auto maskType = cast<RankedTensorType>(iterMask.getType());
    if (!range || range.getStart() != 0 || range.getEnd() <= 0 ||
        range.getEnd() != maskType.getDimSize(0) ||
        !loop.isDefinedOutsideOfLoop(range.getResult()))
      return failure();
    OpFoldResult limit = getPrefixBound(condition, range);
    if (!limit)
      return failure();

    Value init = loop.getInitArgs()[slot];
    OpFoldResult initialBound = getPrefixBound(init, range);
    if (!initialBound) {
      auto constant = init.getDefiningOp<arith::ConstantOp>();
      auto dense = constant
                       ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                       : DenseIntElementsAttr();
      if (!dense || !dense.isSplat())
        return failure();
      initialBound = rewriter.getI32IntegerAttr(
          dense.getSplatValue<APInt>().isZero() ? 0 : range.getEnd());
    }

    IRMapping mapping;
    auto materializeBound = [&](OpFoldResult bound) -> Value {
      if (auto value = dyn_cast<Value>(bound))
        return mapping.lookupOrDefault(value);
      return rewriter.create<arith::ConstantOp>(
          loop.getLoc(), cast<IntegerAttr>(cast<Attribute>(bound)));
    };
    auto makeMask = [&](Value bound) -> Value {
      auto splat = rewriter.create<triton::SplatOp>(loop.getLoc(),
                                                    range.getType(), bound);
      return rewriter.create<arith::CmpIOp>(
          loop.getLoc(), arith::CmpIPredicate::slt, range.getResult(), splat);
    };

    SmallVector<Value> inits(loop.getInitArgs());
    inits[slot] = materializeBound(initialBound);
    auto newLoop = rewriter.create<scf::ForOp>(
        loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), inits);
    newLoop->setAttrs(loop->getAttrs());
    if (!newLoop.getBody()->empty())
      rewriter.eraseOp(newLoop.getBody()->getTerminator());
    rewriter.setInsertionPointToStart(newLoop.getBody());
    mapping.map(loop.getInductionVar(), newLoop.getInductionVar());
    for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
      Value replacement = newLoop.getRegionIterArgs()[index];
      if (index == slot)
        replacement = makeMask(replacement);
      mapping.map(argument, replacement);
    }

    Value nextBound;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (&op == update.getOperation()) {
        nextBound = rewriter.create<arith::MinSIOp>(
            update.getLoc(), newLoop.getRegionIterArgs()[slot],
            materializeBound(limit));
        mapping.map(update.getResult(), makeMask(nextBound));
      } else {
        rewriter.clone(op, mapping);
      }
    }
    SmallVector<Value> yielded;
    for (auto [index, value] : llvm::enumerate(yield.getOperands()))
      yielded.push_back(index == slot ? nextBound
                                      : mapping.lookupOrDefault(value));
    rewriter.create<scf::YieldOp>(yield.getLoc(), yielded);

    rewriter.setInsertionPointAfter(newLoop);
    SmallVector<Value> results(newLoop.getResults());
    // Reconstruct the initial mask too when the loop executes zero times.
    results[slot] = makeMask(results[slot]);
    rewriter.replaceOp(loop, results);
    return success();
  }
};

static bool dependsOnLoopMask(Value mask) {
  if (!mask)
    return false;
  SmallVector<Value> worklist{mask};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    auto type = dyn_cast<RankedTensorType>(value.getType());
    bool isMask = type && type.getElementType().isInteger(1);
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (isMask &&
          isa<LoopLikeOpInterface>(argument.getOwner()->getParentOp()))
        return true;
      continue;
    }
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (isMask && isa<LoopLikeOpInterface>(producer))
      return true;
    llvm::append_range(worklist, producer->getOperands());
    // Follow captured masks through region results, e.g. scf.if yields.
    for (Region &region : producer->getRegions())
      for (Block &block : region)
        llvm::append_range(worklist, block.getTerminator()->getOperands());
  }
  return false;
}

static void markRuntimeLoopMasks(ModuleOp module) {
  module.walk([](Operation *op) {
    Value mask;
    if (auto load = dyn_cast<triton::LoadOp>(op))
      mask = load.getMask();
    else if (auto store = dyn_cast<triton::StoreOp>(op))
      mask = store.getMask();
    else if (auto atomic = dyn_cast<triton::AtomicRMWOp>(op))
      mask = atomic.getMask();
    if (!dependsOnLoopMask(mask))
      return;
    OpBuilder builder(op);
    MaskState state;
    if (succeeded(state.parse(mask, op->getLoc(), builder)))
      return;
    // Unknown loop masks must reach a lowering that consumes the actual mask.
    // In particular, do not use the SIMD full-load-and-select fallback.
    op->setAttr(ConverterUtils::runtimeLoopMaskAttrName, builder.getUnitAttr());
    op->setAttr(ConverterUtils::mixCompileDiscreteMaskAttrName,
                builder.getUnitAttr());
  });
}

static void markSyncBlockLockUnordered(Operation *op) {
  op->setAttr(hivm::SyncBlockLockUnorderedAttr::name,
              UnitAttr::get(op->getContext()));
}

static bool traceUserToTargetOp(Value val) {
  llvm::SmallVector<Value, 32> worklist;
  llvm::SmallPtrSet<Value, 32> visited;
  worklist.push_back(val);

  while (!worklist.empty()) {
    Value currVal = worklist.pop_back_val();
    if (!visited.insert(currVal).second)
      continue;

    for (Operation *user : currVal.getUsers()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(user)) {
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        Value constVal;
        if (lhs.getDefiningOp<arith::ConstantOp>()) {
          constVal = lhs;
        } else if (rhs.getDefiningOp<arith::ConstantOp>()) {
          constVal = rhs;
        } else {
          continue;
        }

        auto constDef = constVal.getDefiningOp<arith::ConstantOp>();
        int64_t blockSize = 0;
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constDef.getValue())) {
          blockSize = intAttr.getInt();
        }

        llvm::SmallVector<Value, 8> searchQueue;
        llvm::SmallPtrSet<Value, 8> searchVis;
        for (Value mulRes : mulOp->getResults()) {
          searchQueue.push_back(mulRes);
          searchVis.insert(mulRes);
        }

        bool findMatch = false;
        while (!searchQueue.empty()) {
          Value checkVal = searchQueue.pop_back_val();
          for (Operation *subUser : checkVal.getUsers()) {
            if (auto addOp = dyn_cast<arith::AddIOp>(subUser)) {
              Value otherOperand = (addOp.getLhs() == checkVal)
                                       ? addOp.getRhs()
                                       : addOp.getLhs();

              Value curSrc = otherOperand;
              bool hitRange = false;
              int depth = 0;
              while (curSrc.getDefiningOp() && depth < 5) {
                Operation *defOp = curSrc.getDefiningOp();
                if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(defOp)) {
                  if (rangeOp.getEnd() == blockSize) {
                    hitRange = true;
                    break;
                  }
                }

                if (isa<arith::ExtSIOp, triton::SplatOp, triton::ExpandDimsOp,
                        triton::BroadcastOp>(defOp)) {
                  curSrc = defOp->getOperand(0);
                  depth++;
                  continue;
                }
                break;
              }
              if (hitRange) {
                findMatch = true;
                break;
              }
            }
            if (isa<arith::ExtSIOp, triton::SplatOp, triton::ExpandDimsOp,
                    triton::BroadcastOp>(subUser)) {
              for (Value subRes : subUser->getResults()) {
                if (!searchVis.count(subRes)) {
                  searchVis.insert(subRes);
                  searchQueue.push_back(subRes);
                }
              }
            }
          }
          if (findMatch)
            break;
        }
        if (findMatch) {
          return true;
        }

        for (Value mulRes : mulOp->getResults()) {
          worklist.push_back(mulRes);
        }
      }

      if (isa<arith::ExtSIOp, triton::SplatOp, triton::ExpandDimsOp,
              triton::BroadcastOp>(user)) {
        for (Value res : user->getResults()) {
          worklist.push_back(res);
        }
      }
    }
  }
  return false;
}

static bool checkAllProgramIdNonOverlap(ModuleOp module) {
  bool allNonOverlap = true;
  module.walk([&](triton::GetProgramIdOp pidOp) {
    if (!traceUserToTargetOp(pidOp.getResult())) {
      allNonOverlap = false;
    }
  });
  return allNonOverlap;
}

LogicalResult isDiscreteMask(Operation *op, Value mask,
                             PatternRewriter &rewriter) {
  if (!mask || op->hasAttr(ConverterUtils::mixCompileDiscreteMaskAttrName)) {
    return failure();
  }

  MaskState mstate;
  auto isContMask = mstate.parse(mask, op->getLoc(), rewriter);
  if (!isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }
  return success();
}

// Recursively collect all leaf operands of a nested arith::AndIOp tree.
// This function also normalizes masks by distributing broadcast over andi
//   broadcast(andi(a, b)) = andi(broadcast(a), broadcast(b))
// so that inner AND operands nested inside a broadcast are still reachable.
static void collectAndLeaves(Value mask, SmallVectorImpl<Value> &leaves,
                             Location loc, PatternRewriter &rewriter) {
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    collectAndLeaves(andOp.getLhs(), leaves, loc, rewriter);
    collectAndLeaves(andOp.getRhs(), leaves, loc, rewriter);
  } else if (auto broadcastOp = mask.getDefiningOp<triton::BroadcastOp>()) {
    // Distribute broadcast over andi so we can inspect each factor separately.
    if (auto innerAnd = broadcastOp.getSrc().getDefiningOp<arith::AndIOp>()) {
      Type dstType = mask.getType();
      Value broadcastA =
          rewriter.create<triton::BroadcastOp>(loc, dstType, innerAnd.getLhs())
              .getResult();
      Value broadcastB =
          rewriter.create<triton::BroadcastOp>(loc, dstType, innerAnd.getRhs())
              .getResult();
      collectAndLeaves(broadcastA, leaves, loc, rewriter);
      collectAndLeaves(broadcastB, leaves, loc, rewriter);
    } else {
      leaves.push_back(mask);
    }
  } else {
    leaves.push_back(mask);
  }
}

struct MaskDecomposition {
  // AND of all leaves that MaskState::parse() can analyze as a rectangle mask.
  // nullptr when no such leaves exist.
  Value contMask;
  // AND of all leaves that MaskState::parse() cannot analyze
  // (discrete/runtime). nullptr when no such leaves exist.
  Value discMask;
};

// Decompose an AND-tree mask into its continuous and discrete leaf components
// so that we can use contMask to bound GM accesses while discMask still drives
// the per-element selection.
static MaskDecomposition decomposeAndMask(Operation *op, Value mask,
                                          const Location &loc,
                                          PatternRewriter &rewriter) {
  SmallVector<Value> leaves;
  collectAndLeaves(mask, leaves, loc, rewriter);

  SmallVector<Value> contLeaves;
  SmallVector<Value> discLeaves;

  for (Value leaf : leaves) {
    MaskState st;
    if (st.parse(leaf, loc, rewriter).succeeded()) {
      if (st.isMask())
        contLeaves.push_back(leaf);
      else
        discLeaves.push_back(leaf);
    } else {
      discLeaves.push_back(leaf);
    }
  }

  Value contMask = nullptr;
  for (Value v : contLeaves)
    contMask =
        contMask ? rewriter.create<arith::AndIOp>(loc, contMask, v).getResult()
                 : v;

  Value discMask = nullptr;
  for (Value v : discLeaves)
    discMask =
        discMask ? rewriter.create<arith::AndIOp>(loc, discMask, v).getResult()
                 : v;

  return {contMask, discMask};
}

struct DiscreteMaskStoreConversion : OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const final {
    if (op->hasAttr(ConverterUtils::mixCompileDiscreteMaskAttrName))
      return failure();

    auto mask = op.getMask();
    auto loc = op.getLoc();
    auto dst = op.getPtr();
    auto src = op.getValue();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    auto ptr = op.getPtr();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    bool rankWithinIndirectFastPathLimit =
        ptrType && ptrType.getShape().size() <= 5;
    if (compileOn91095Flag &&
        triton::ascend::isSimtTemplateMode(compileModeFlag) &&
        rankWithinIndirectFastPathLimit) {
      op->setAttr(ConverterUtils::mixCompileDiscreteMaskAttrName,
                  rewriter.getUnitAttr());
      return failure();
    }

    // When mask = contMask & discMask, use contMask to bound GM accesses and
    // discMask to select the final per-element value. This prevents the
    // unguarded full-load from reading past the tail-block boundary.
    auto [contMask, discMask] = decomposeAndMask(op, mask, loc, rewriter);
    if (contMask && discMask) {
      // insert sync_block_lock (unordered: see markSyncBlockLockUnordered)
      auto lockVar = MemOpConverter::createSyncBlockLockVar(rewriter, loc);
      markSyncBlockLockUnordered(lockVar.getOperation());
      if (useSyncBlockLockFlag) {
        rewriter.create<hivm::PipeBarrierOp>(
            loc,
            hivm::PipeAttr::get(rewriter.getContext(), hivm::PIPE::PIPE_ALL));
        auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
        markSyncBlockLockUnordered(lockOp.getOperation());
      }
      auto safeLoad = rewriter.create<triton::LoadOp>(
          loc, dst, contMask, op.getCache(), op.getEvict(), false);
      auto selOp = rewriter.create<arith::SelectOp>(loc, discMask, src,
                                                    safeLoad.getResult());
      auto newStore = rewriter.create<triton::StoreOp>(
          loc, dst, selOp, contMask, op.getCache(), op.getEvict());
      newStore->setAttr(ConverterUtils::discreteMaskAttrName,
                        UnitAttr::get(rewriter.getContext()));
      if (useSyncBlockLockFlag) {
        auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
        markSyncBlockLockUnordered(unlockOp.getOperation());
      }
      rewriter.replaceOp(op, newStore);
      return success();
    }

    // SIMD fallback: original full load + select (contMask absent, pure
    // discrete). Has DDR OOB risk but no better option in pure simd mode.
    // insert sync_block_lock to serialize the read-modify-write window.
    auto lockVar = MemOpConverter::createSyncBlockLockVar(rewriter, loc);
    markSyncBlockLockUnordered(lockVar.getOperation());
    if (useSyncBlockLockFlag) {
      rewriter.create<hivm::PipeBarrierOp>(
          loc,
          hivm::PipeAttr::get(rewriter.getContext(), hivm::PIPE::PIPE_ALL));
      auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
      markSyncBlockLockUnordered(lockOp.getOperation());
    }
    auto loadFromDstOp = rewriter.create<triton::LoadOp>(
        loc, dst, op.getCache(), op.getEvict(), false);
    auto selOp = rewriter.create<arith::SelectOp>(loc, mask, src,
                                                  loadFromDstOp.getResult());
    auto newStore = rewriter.create<triton::StoreOp>(
        loc, dst, selOp, op.getCache(), op.getEvict());
    newStore->setAttr(ConverterUtils::discreteMaskAttrName,
                      UnitAttr::get(rewriter.getContext()));
    if (useSyncBlockLockFlag) {
      auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
      markSyncBlockLockUnordered(unlockOp.getOperation());
    }
    rewriter.replaceOp(op, newStore);
    return success();
  }
};

struct DiscreteMaskLoadConversion : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const final {
    if (op->hasAttr(ConverterUtils::mixCompileDiscreteMaskAttrName))
      return failure();

    auto loc = op.getLoc();
    auto other = op.getOther();
    auto mask = op.getMask();
    auto ptr = op.getPtr();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    bool rankWithinIndirectFastPathLimit =
        ptrType && ptrType.getShape().size() <= 5;
    if (compileOn91095Flag &&
        triton::ascend::isSimtTemplateMode(compileModeFlag) &&
        rankWithinIndirectFastPathLimit) {
      op->setAttr(ConverterUtils::mixCompileDiscreteMaskAttrName,
                  rewriter.getUnitAttr());
      return failure();
    }

    // SIMD path: when mask = contMask & discMask, load only the safe range
    // defined by contMask and use discMask for the per-element select,
    // avoiding OOB reads.
    auto [contMask, discMask] = decomposeAndMask(op, mask, loc, rewriter);
    if (contMask && discMask) {
      if (!other) {
        FailureOr<Value> constant = specializeTypelessValueToConstant(
            TypelessValue::Zero, ptr.getType(), loc, rewriter);
        if (failed(constant)) {
          llvm_unreachable("Unsupported type for constant creation");
        }
        other = *constant;
      }
      auto safeLoad = rewriter.create<triton::LoadOp>(
          loc, ptr, contMask, op.getCache(), op.getEvict(), op.getIsVolatile());
      // Use combined mask to select the result, avoid the uninitialized memory
      // access.
      auto combinedMask =
          rewriter.create<arith::AndIOp>(loc, contMask, discMask);
      auto discreteMaskOp = rewriter.create<arith::SelectOp>(
          loc, combinedMask, safeLoad.getResult(), other);
      rewriter.replaceOp(op, discreteMaskOp);
      return success();
    }

    // Fallback: original full load + select (contMask absent, pure discrete).
    if (!other) {
      FailureOr<Value> constant = specializeTypelessValueToConstant(
          TypelessValue::Zero, ptr.getType(), loc, rewriter);
      if (failed(constant))
        llvm_unreachable("Unsupported type for constant creation");
      other = *constant;
    }

    auto newLoadOp = rewriter.create<triton::LoadOp>(
        loc, ptr, op.getCache(), op.getEvict(), op.getIsVolatile());
    auto discreteMaskOp =
        rewriter.create<arith::SelectOp>(loc, mask, newLoadOp, other);
    rewriter.replaceOp(op, discreteMaskOp);
    return success();
  }
};

struct DiscreteMaskAtomicConversion
    : OpRewritePattern<mlir::triton::AtomicRMWOp> {
  using OpRewritePattern<mlir::triton::AtomicRMWOp>::OpRewritePattern;

  bool routeToSIMD(mlir::triton::AtomicRMWOp op,
                   PatternRewriter &rewriter) const {
    if (op->hasAttr(ConverterUtils::explicitRouteToSIMDAttrName)) {
      return op
          ->getAttrOfType<BoolAttr>(ConverterUtils::explicitRouteToSIMDAttrName)
          .getValue();
    }

    auto loc = op.getLoc();
    auto ptr = op.getPtr();
    llvm::DenseMap<Value, PtrOffsetInfo> offsetMap;
    triton::parse(ptr, loc, rewriter, offsetMap);
    auto it = offsetMap.find(ptr);
    auto toSimd = it != offsetMap.end() &&
                  llvm::all_of(it->second.getStructuredRef(), [](auto dim) {
                    return dim == PtrOffsetInfo::AxisInfo::structured;
                  });
    op->setAttr(ConverterUtils::explicitRouteToSIMDAttrName,
                rewriter.getBoolAttr(toSimd));
    return toSimd;
  }

  LogicalResult matchAndRewrite(mlir::triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptr = op.getPtr();
    auto src = op.getVal();
    auto mask = op.getMask();
    RMWOp rmwOp = op.getAtomicRmwOp();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    // Keep the original mask for the SIMT atomic template when the pointer
    // needs the indirect atomic ABI. The unstructure pass flattens it together
    // with offsets and values; replacing it with a select here would lose that
    // lane mask. Structured pointers do not need the indirect ABI, so fall
    // through to the SIMD select+atomic rewrite below instead.
    if (compileOn91095Flag &&
        triton::ascend::isSimtTemplateMode(compileModeFlag) &&
        !routeToSIMD(op, rewriter)) {
      op->setAttr(ConverterUtils::mixCompileDiscreteMaskAttrName,
                  rewriter.getUnitAttr());
      return failure();
    }

    const std::map<RMWOp, TypelessValue> initMap = {
        {RMWOp::FADD, TypelessValue::Zero},
        {RMWOp::ADD, TypelessValue::Zero},
        {RMWOp::UMAX, TypelessValue::Zero},
        {RMWOp::OR, TypelessValue::Zero},
        {RMWOp::MIN, TypelessValue::Max},
        {RMWOp::UMIN, TypelessValue::Max},
        {RMWOp::AND, TypelessValue::Max},
        {RMWOp::MAX, TypelessValue::Min},
        {RMWOp::XOR, TypelessValue::Zero},
        {RMWOp::XCHG, TypelessValue::Undefined},
    };
    assert(initMap.find(rmwOp) != initMap.end());
    auto typelessVal = initMap.at(rmwOp);
    if (typelessVal == TypelessValue::Undefined) {
      // Undefined default value atomic op will be decomposed in AscendNPU-IR
      op->setAttr(ConverterUtils::discreteMaskAttrName,
                  UnitAttr::get(rewriter.getContext()));
      return failure();
    }

    auto [contMask, discMask] = decomposeAndMask(op, mask, loc, rewriter);
    FailureOr<mlir::Value> fill = specializeTypelessValueToConstant(
        typelessVal, src.getType(), loc, rewriter);
    if (failed(fill)) {
      LLVM_DEBUG({
        llvm::dbgs() << "Unsupported type for constant creation: "
                     << src.getType() << "\n";
      });
      op->emitError("Unsupported atomic operation.");
      return failure();
    }

    // For mask = contMask & discMask, retain contMask as the memory-access
    // guard and replace only the discrete part with the RMW identity value.
    // The continuous mask can then be lowered to a bounded subview without
    // accessing the masked-off tail.
    Value valueMask = mask;
    Value accessMask = nullptr;
    if (contMask && discMask) {
      valueMask = discMask;
      accessMask = contMask;
    }

    auto maskedValue =
        rewriter.create<arith::SelectOp>(loc, valueMask, src, *fill);
    auto newAtomicOp = rewriter.create<mlir::triton::AtomicRMWOp>(
        loc, src.getType(), rmwOp, ptr, maskedValue, accessMask, op.getSem(),
        op.getScope());
    rewriter.replaceOp(op, newAtomicOp);
    return success();
  }
};

DiscreteMaskAccessConversionPass::DiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options)
    : DiscreteMaskAccessConversionBase(options) {}

void DiscreteMaskAccessConversionPass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;
  auto compileMode = triton::ascend::parseCompileMode(this->compileMode);
  if (!compileMode) {
    getOperation().emitError()
        << "discrete-mask-access-conversion compile-mode is invalid: "
        << this->compileMode;
    signalPassFailure();
    return;
  }
  compileModeFlag = *compileMode;
  auto moduleOp = getOperation();
  RewritePatternSet loopMaskPatterns(&getContext());
  loopMaskPatterns.add<CanonicalizeLoopPrefixMask>(&getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(loopMaskPatterns)))) {
    moduleOp.emitError("failed to canonicalize loop prefix masks");
    signalPassFailure();
    return;
  }
  markRuntimeLoopMasks(moduleOp);

  bool tileNonOverlap = checkAllProgramIdNonOverlap(moduleOp);
  useSyncBlockLockFlag = !tileNonOverlap;

  // Restore floating-point atomic max/min expanded by semantic.py before
  // discrete-mask rewriting changes the atomic value into arith.select.
  // Run this in a separate greedy-rewrite phase so that
  // DiscreteMaskAtomicConversion cannot consume the expanded form first.
  RewritePatternSet atomicMaxMinPatterns(&getContext());
  atomicMaxMinPatterns.add<LoadStoreConverter::AtomicMaxMinCanonicalizer>(
      atomicMaxMinPatterns.getContext());
  if (failed(
          applyPatternsGreedily(moduleOp, std::move(atomicMaxMinPatterns)))) {
    moduleOp->emitError("failed to canonicalize floating-point atomic max/min");
    signalPassFailure();
    return;
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<DiscreteMaskLoadConversion, DiscreteMaskStoreConversion,
               DiscreteMaskAtomicConversion>(patterns.getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply discrete mask access patterns");
    signalPassFailure();
  }

  // Clean up dead analysis ops left behind by MaskState::parse().
  // These are trivially-dead auxiliary ops (constants, arithmetic) with no
  // users that parse() creates as side effects of mask analysis.
  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    moduleOp->emitWarning(
        "DiscreteMaskAccessConversion: dead-code cleanup failed");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "==============================================\n";
    llvm::dbgs() << "After DiscreteMaskAccessConversionPass:\n" << moduleOp;
    llvm::dbgs() << "\n==============================================\n";
  });
}

void DiscreteMaskAccessConversionPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect, scf::SCFDialect, triton::TritonDialect,
                  hivm::HIVMDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options) {
  return std::make_unique<DiscreteMaskAccessConversionPass>(options);
}
