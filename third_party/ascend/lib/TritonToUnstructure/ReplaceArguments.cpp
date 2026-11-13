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

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "Utils/Utils.h"

#include "mlir/IR/Block.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-replace-arguments"

using namespace mlir;
using namespace triton;

namespace {

inline constexpr llvm::StringLiteral kScalarPointerCarrierBoundaryAttr =
    "ScalarPointerCarrierBoundary";

static bool isScalarPointerType(Type type) {
  auto pointerType = dyn_cast<triton::PointerType>(type);
  return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
}

static bool isPointerDescriptorBoundarySlot(Operation *loop, unsigned slot) {
  if (!loop)
    return false;
  auto slots = dyn_cast_or_null<DenseI32ArrayAttr>(
      loop->getAttr(controlflow::kPointerDescriptorBoundaryAttr));
  return slots &&
         llvm::is_contained(slots.asArrayRef(), static_cast<int32_t>(slot));
}

// Only scalar pointers owned by the CFO descriptor schema cross a loop as a
// complete i64 address. Ordinary scalar pointers retain the established
// base-plus-relative-offset representation.
static bool useCompleteScalarAddress(Operation *loop, unsigned slot,
                                     Type type) {
  return loop && isa<scf::ForOp, scf::WhileOp>(loop) &&
         isScalarPointerType(type) &&
         (isPointerDescriptorBoundarySlot(loop, slot) ||
          loop->hasAttr(kScalarPointerCarrierBoundaryAttr));
}

// Record the exact loop slot whose block argument is changed from a scalar
// pointer to an i64 relative offset.  The marker is deliberately attached to
// the cloned SCF op rather than inferred later from type alone: ordinary i64
// loop-carried values must retain their established analysis.
static void markScalarPointerOffsetSlot(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return;
  Operation *owner = blockArg.getOwner()->getParentOp();
  if (!owner)
    return;

  unsigned slot = 0;
  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    if (blockArg.getOwner() != forOp.getBody() || blockArg.getArgNumber() == 0)
      return;
    slot = blockArg.getArgNumber() - 1;
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(owner)) {
    if (blockArg.getOwner() != whileOp.getBeforeBody() &&
        blockArg.getOwner() != whileOp.getAfterBody())
      return;
    slot = blockArg.getArgNumber();
  } else {
    return;
  }

  SmallVector<int32_t> slots;
  if (auto existing = dyn_cast_or_null<DenseI32ArrayAttr>(
          owner->getAttr(kScalarPointerOffsetBoundaryAttr)))
    slots.append(existing.asArrayRef().begin(), existing.asArrayRef().end());
  if (!llvm::is_contained(slots, static_cast<int32_t>(slot)))
    slots.push_back(static_cast<int32_t>(slot));
  owner->setAttr(kScalarPointerOffsetBoundaryAttr,
                 DenseI32ArrayAttr::get(owner->getContext(), slots));
}

// A scalar-pointer SCF slot is represented by one complete i64 address on all
// structural edges. Region-local pointer users are rebuilt immediately, so no
// pointer type crosses the control-flow boundary.
static Value rebuildScalarPointer(Value address, Type pointerType,
                                  OpBuilder &builder, Location loc) {
  return builder.create<triton::IntToPtrOp>(loc, pointerType, address);
}

static Value materializeScalarPointerAddress(Value pointer, OpBuilder &builder,
                                             Location loc) {
  return builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(), pointer);
}

static bool hasScalarPointerBase(const PtrOffsetInfo &info) {
  return info.getPtr() && isScalarPointerType(info.getPtr().getType());
}

// A function argument is a stable pointer base: it dominates the enclosing
// loop and carries no control-flow-selected address.  Offset analysis records
// this base with `ptr == value` and a zero displacement, so that identity is
// valid for a loop init but must not be confused with a loop phi or an
// int_to_ptr result elsewhere.
static bool isStableFunctionScalarPointerBase(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || !isScalarPointerType(value.getType()))
    return false;
  Operation *owner = blockArg.getOwner()->getParentOp();
  return owner && isa<FunctionOpInterface>(owner);
}

// Offset analysis deliberately represents a scalar pointer selected by
// scf.if as an opaque complete address. If such a value is a loop backedge,
// the loop must choose the complete-address protocol before any edge is
// rewritten; otherwise the init/region argument can become i64 while the
// yield remains a pointer.
static bool isOpaqueScalarPointerIfResult(Value value) {
  return isScalarPointerType(value.getType()) &&
         value.getDefiningOp<scf::IfOp>();
}

// A scalar pointer can use the established T2U offset representation only if
// analysis found both a source pointer and a displacement.  An entry whose
// source is the value itself is normally an opaque complete address (for
// example a control-flow phi).  A function argument is the one exception when
// it is examined as the init anchor of an ordinary loop; the loop conversion
// must then materialize a zero offset for that anchor.
static bool
canRewriteScalarPointer(Value value, RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (!isScalarPointerType(value.getType()))
    return true;

  parse(value, value.getLoc(), rewriter, offsetMap);
  auto it = offsetMap.find(value);
  if (it == offsetMap.end())
    return false;
  const PtrOffsetInfo &info = it->second;
  return info.getPtr() && info.getOffset() && info.getPtr() != value;
}

// Decide the representation once for the complete SCF boundary.  The caller
// passes this same decision to every init, region argument, yield, condition
// operand, and result rewrite.  Thus a failed proof falls back atomically to
// the pointer representation instead of producing a mixed-type boundary.
static bool
shouldPreserveScalarPointers(Operation *op, RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto canRewriteBoundary = [&](Value value, bool allowStableBase) {
    if (!isScalarPointerType(value.getType()))
      return true;
    if (allowStableBase && isStableFunctionScalarPointerBase(value))
      return true;
    return canRewriteScalarPointer(value, rewriter, offsetMap);
  };

  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    for (Value init : whileOp.getInits())
      if (!canRewriteBoundary(init, /*allowStableBase=*/true))
        return true;
    for (Value arg : whileOp.getBeforeArguments())
      if (!canRewriteBoundary(arg, /*allowStableBase=*/false))
        return true;
    for (Value arg : whileOp.getAfterArguments())
      if (!canRewriteBoundary(arg, /*allowStableBase=*/false))
        return true;
    for (Value value : whileOp.getYieldOp().getOperands())
      if (isOpaqueScalarPointerIfResult(value))
        return true;
    return false;
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    for (Value init : loopOp.getInits())
      if (!canRewriteBoundary(init, /*allowStableBase=*/true))
        return true;
    for (Value arg : loopOp.getRegionIterArgs())
      if (!canRewriteBoundary(arg, /*allowStableBase=*/false))
        return true;
    for (Value value : loopOp.getYieldedValues())
      if (isOpaqueScalarPointerIfResult(value))
        return true;
    return false;
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (Value value : ifOp->getResults())
      if (!canRewriteBoundary(value, /*allowStableBase=*/false))
        return true;
    for (Value value : ifOp.thenYield().getResults())
      if (!canRewriteBoundary(value, /*allowStableBase=*/false))
        return true;
    if (ifOp.elseBlock())
      for (Value value : ifOp.elseYield().getResults())
        if (!canRewriteBoundary(value, /*allowStableBase=*/false))
          return true;
  } else {
    return false;
  }
  return false;
}

static bool hasScalarPointerResult(scf::IfOp op) {
  return llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isScalarPointerType(type); });
}

static bool hasScalarPointerBoundary(Operation *op) {
  SmallVector<Value> values;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    values.append(whileOp.getInits().begin(), whileOp.getInits().end());
    values.append(whileOp.getBeforeArguments().begin(),
                  whileOp.getBeforeArguments().end());
    values.append(whileOp.getAfterArguments().begin(),
                  whileOp.getAfterArguments().end());
    values.append(whileOp->getResults().begin(), whileOp->getResults().end());
    values.append(whileOp.getConditionOp().getArgs().begin(),
                  whileOp.getConditionOp().getArgs().end());
    values.append(whileOp.getYieldOp()->getOperands().begin(),
                  whileOp.getYieldOp()->getOperands().end());
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    values.append(forOp.getInitArgs().begin(), forOp.getInitArgs().end());
    values.append(forOp.getRegionIterArgs().begin(),
                  forOp.getRegionIterArgs().end());
    values.append(forOp->getResults().begin(), forOp->getResults().end());
    values.append(forOp.getYieldedValues().begin(),
                  forOp.getYieldedValues().end());
  } else {
    return false;
  }
  return llvm::any_of(
      values, [](Value value) { return isScalarPointerType(value.getType()); });
}

static void markOpaqueScalarPointerBoundary(Operation *op) {
  if (hasScalarPointerBoundary(op))
    op->setAttr(kScalarPointerCarrierBoundaryAttr,
                UnitAttr::get(op->getContext()));
}

// `shouldPreserveScalarPointers` analyzes every value on the SCF boundary to
// choose one representation.  When the choice is relative offsets, those
// cached entries describe the old pointer-typed boundary and must be removed
// before replaceArgs/replaceOperands rebuild the boundary.  Keeping even one
// stale yield or result entry is enough to bypass the live backedge analysis.
static void invalidateScalarPointerBoundaryAnalysis(
    Operation *op, llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto eraseScalarPointers = [&](ValueRange values) {
    for (Value value : values)
      if (isScalarPointerType(value.getType()))
        offsetMap.erase(value);
  };

  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    eraseScalarPointers(whileOp.getInits());
    eraseScalarPointers(whileOp.getBeforeArguments());
    eraseScalarPointers(whileOp.getAfterArguments());
    eraseScalarPointers(whileOp->getResults());
    eraseScalarPointers(whileOp.getConditionOp().getArgs());
    eraseScalarPointers(whileOp.getYieldOp()->getOperands());
    return;
  }
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    eraseScalarPointers(loopOp.getInits());
    eraseScalarPointers(loopOp.getRegionIterArgs());
    eraseScalarPointers(loopOp->getResults());
    eraseScalarPointers(loopOp.getYieldedValues());
    return;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    eraseScalarPointers(ifOp->getResults());
    eraseScalarPointers(ifOp.thenYield().getResults());
    if (ifOp.elseBlock())
      eraseScalarPointers(ifOp.elseYield().getResults());
  }
}

} // namespace

void replaceOperands(MutableArrayRef<OpOperand> oprs, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                     bool preserveScalarPointers,
                     bool allowStableBaseInit = false) {
  for (auto it = oprs.begin(); it != oprs.end(); ++it) {
    auto &opr = *it;
    auto operand = opr.get();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      const PtrOffsetInfo &info = offsetMap.at(operand);
      // A complete opaque tensor base cannot be represented as an offset from
      // one scalar source. Keep this structural edge unchanged instead of
      // creating a type-mismatched loop signature.
      if (hasScalarPointerBase(info))
        opr.set(info.getOffset());
    } else if (auto ptrType =
                   dyn_cast<triton::PointerType>(operand.getType())) {
      // An unmarked scalar pointer has no complete structural carrier schema.
      // Keep it as a pointer on every SCF edge instead of changing only this
      // operand to a relative offset and invalidating the parent operation.
      if (preserveScalarPointers && isScalarPointerType(operand.getType()))
        continue;
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      if (allowStableBaseInit && isStableFunctionScalarPointerBase(operand)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(opr.getOwner());
        Value zero =
            rewriter.create<arith::ConstantIntOp>(operand.getLoc(), 0, 64);
        opr.set(zero);
        continue;
      }
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        for (auto offset : offsetMap.at(operand).getOffsets()) {
          it->set(offset);
          ++it;
        }
        --it;
      } else {
        // source == operand marks a complete scalar address whose source may be
        // selected or loop-carried at runtime. Keep it as a pointer instead of
        // replacing it with an offset relative to one statically chosen base.
        if (offsetMap.at(operand).getPtr() == operand)
          continue;
        opr.set(offsetMap.at(operand).getOffset());
      }
    }
  }
}

void replaceArgs(ValueRange args, RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                 bool preserveScalarPointers) {
  for (auto it = args.begin(); it != args.end(); ++it) {
    auto arg = *it;
    if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      const PtrOffsetInfo &info = offsetMap.at(arg);
      // The relative-offset representation is legal only with one scalar
      // element-pointer base. Opaque tensor bases remain pointer values and
      // are handled lane-wise (or rejected later by the boundary validator).
      if (!hasScalarPointerBase(info))
        continue;

      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                         .create<UnrealizedConversionCastOp>(
                             arg.getLoc(), arg.getType(), ValueRange({}))
                         ->getResult(0);
      Value src = info.getPtr();
      rewriter.replaceAllUsesWith(arg, tempVar);
      arg.setType(RankedTensorType::get(tensorType.getShape(),
                                        rewriter.getIntegerType(64)));
      src = rewriter.create<triton::SplatOp>(arg.getLoc(), tempVar.getType(),
                                             src);
      rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
          tempVar.getDefiningOp(), tempVar.getType(), src, arg);
    } else if (auto ptrType = dyn_cast<triton::PointerType>(arg.getType())) {
      // Region arguments and results must use the same representation as the
      // corresponding init/condition/yield operands.
      if (preserveScalarPointers && isScalarPointerType(arg.getType()))
        continue;
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      if (!isa<RankedTensorType>(ptrType.getPointeeType()) &&
          offsetMap.at(arg).getPtr() == arg)
        continue;

      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                         .create<UnrealizedConversionCastOp>(
                             arg.getLoc(), arg.getType(), ValueRange({}))
                         ->getResult(0);
      rewriter.replaceAllUsesWith(arg, tempVar);
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        auto srcOp =
            offsetMap.at(arg).getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        arg.setType(rewriter.getIntegerType(32));
        SmallVector<Value> newOffsets;
        for (auto offset : offsetMap.at(arg).getOffsets()) {
          newOffsets.push_back(*it);
          ++it;
        }
        --it;
        rewriter.replaceOpWithNewOp<triton::MakeTensorPtrOp>(
            tempVar.getDefiningOp(), tempVar.getType(), srcOp.getBase(),
            srcOp.getShape(), srcOp.getStrides(), newOffsets, srcOp.getOrder());
      } else {
        auto src = offsetMap.at(arg).getPtr();
        offsetMap.erase(arg);
        arg.setType(rewriter.getIntegerType(64));
        rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
            tempVar.getDefiningOp(), tempVar.getType(), src, arg);
      }
    }
  }
}

void convertTensorPtrPre(Operation *op, RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Preorder start\n" << *op << "\n";
  });
  bool preserveScalarPointers =
      shouldPreserveScalarPointers(op, rewriter, offsetMap);
  if (!preserveScalarPointers)
    invalidateScalarPointerBoundaryAnalysis(op, offsetMap);
  SmallVector<Value> scalarPointerOffsetArgs;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    if (!preserveScalarPointers)
      llvm::copy_if(whileOp.getBeforeArguments(),
                    std::back_inserter(scalarPointerOffsetArgs), [](Value arg) {
                      return isScalarPointerType(arg.getType());
                    });
    replaceArgs(whileOp.getBeforeArguments(), rewriter, offsetMap,
                preserveScalarPointers);
    replaceOperands(whileOp.getInitsMutable(), rewriter, offsetMap,
                    preserveScalarPointers,
                    /*allowStableBaseInit=*/true);
    replaceArgs(whileOp.getAfterArguments(), rewriter, offsetMap,
                preserveScalarPointers);
    replaceArgs(whileOp->getResults(), rewriter, offsetMap,
                preserveScalarPointers);
    replaceOperands(whileOp.getConditionOp().getArgsMutable(), rewriter,
                    offsetMap, preserveScalarPointers);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    if (!preserveScalarPointers)
      llvm::copy_if(loopOp.getRegionIterArgs(),
                    std::back_inserter(scalarPointerOffsetArgs), [](Value arg) {
                      return isScalarPointerType(arg.getType());
                    });
    replaceArgs(loopOp.getRegionIterArgs(), rewriter, offsetMap,
                preserveScalarPointers);
    replaceOperands(loopOp.getInitsMutable(), rewriter, offsetMap,
                    preserveScalarPointers,
                    /*allowStableBaseInit=*/true);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (preserveScalarPointers && hasScalarPointerResult(ifOp))
      ifOp->setAttr(kScalarPointerCarrierBoundaryAttr,
                    UnitAttr::get(ifOp.getContext()));
    replaceArgs(ifOp->getResults(), rewriter, offsetMap,
                preserveScalarPointers);
    replaceOperands(ifOp.thenYield().getResultsMutable(), rewriter, offsetMap,
                    preserveScalarPointers);
    replaceOperands(ifOp.elseYield().getResultsMutable(), rewriter, offsetMap,
                    preserveScalarPointers);
  }
  // Publish the offset-carrier schema only after every structural edge has
  // been rewritten. In particular, a While before argument and its paired
  // after argument must both retain their original base provenance while they
  // are converted. Publishing after the first region would make parsing the
  // second region treat its still-pointer-typed argument as a base-less i64
  // offset and attempt to rebuild tt.addptr from a null source.
  for (Value arg : scalarPointerOffsetArgs) {
    markScalarPointerOffsetSlot(arg);
    // replaceArgs() may have cached the old pointer provenance before the
    // marker was published. Drop only the marked region arguments so the
    // subsequent body walk re-parses them as live scalar offset carriers.
    offsetMap.erase(arg);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Preorder end\n" << *op << "\n";
  });
}

void convertTensorPtrPost(Operation *op, RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Postorder start\n" << *op << "\n";
  });
  bool preserveScalarPointers =
      shouldPreserveScalarPointers(op, rewriter, offsetMap);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    replaceOperands(whileOp.getYieldOp()->getOpOperands(), rewriter, offsetMap,
                    preserveScalarPointers);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    replaceArgs(loopOp->getResults(), rewriter, offsetMap,
                preserveScalarPointers);
    replaceOperands(*loopOp.getYieldedValuesMutable(), rewriter, offsetMap,
                    preserveScalarPointers);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Postorder end\n" << *op << "\n";
  });
}

int getPtrTensorRank(Type type) {
  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      return tensorType.getRank();
    }
  }
  return 0;
}

SmallVector<Value> constructOperands(ValueRange operands, Value tempVar,
                                     IRMapping mapping, OpBuilder &builder,
                                     Operation *loop) {
  SmallVector<Value> newOperands;
  for (auto [slot, originalOperand] : llvm::enumerate(operands)) {
    Value mappedOperand = mapping.lookupOrDefault(originalOperand);
    if (useCompleteScalarAddress(loop, slot, originalOperand.getType()))
      mappedOperand = materializeScalarPointerAddress(mappedOperand, builder,
                                                      originalOperand.getLoc());
    newOperands.push_back(mappedOperand);
    auto numAppend = getPtrTensorRank(originalOperand.getType()) - 1;
    if (numAppend > 0)
      newOperands.append(numAppend, tempVar);
  }
  return newOperands;
}

SmallVector<Type> constructTypes(TypeRange types, Operation *loop) {
  SmallVector<Type> newTypes;
  for (auto [slot, type] : llvm::enumerate(types)) {
    newTypes.push_back(useCompleteScalarAddress(loop, slot, type)
                           ? IntegerType::get(type.getContext(), 64)
                           : type);
    if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        if (tensorType.getRank() > 0)
          newTypes.append(tensorType.getRank() - 1,
                          IntegerType::get(type.getContext(), 32));
      }
    }
  }
  return newTypes;
}

void replacePtrArguments(triton::FuncOp funcOp,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Keep replaced values alive until the analysis cache is no longer used.
  // This detached block owns the retired operations without revisiting them.
  Block retiredOps;
  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  Value tempVar = rewriter
                      .create<UnrealizedConversionCastOp>(
                          funcOp.getLoc(), rewriter.getI32Type(), ValueRange{})
                      ->getResult(0);
  std::function<WalkResult(Operation *)> convertTensorPtr = [&](Operation *op) {
    IRMapping mapping;
    Operation *newOp = nullptr;
    // Address carriers used by the replacement op must dominate it. Build the
    // replacement immediately before the old op and unlink the old op last.
    rewriter.setInsertionPoint(op);
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // An opaque scalar-pointer boundary cannot be represented by the
      // relative-offset protocol: one edge may come from an if/select and
      // therefore has no single static base.  Convert every scalar pointer
      // slot to a complete i64 address before constructing the replacement
      // loop, so the loop signature is pointer-free and T2L never has to
      // materialize a pointer from an unresolved SCF value.
      if (shouldPreserveScalarPointers(forOp.getOperation(), rewriter,
                                       offsetMap))
        markOpaqueScalarPointerBoundary(forOp.getOperation());
      SmallVector<Value> newInitArgs = constructOperands(
          forOp.getInitArgs(), tempVar, mapping, rewriter, forOp);
      newOp = rewriter.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newInitArgs,
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            mapping.map(forOp.getInductionVar(), iv);
            auto newArgIter = args.begin();
            for (auto [slot, oldArg] :
                 llvm::enumerate(forOp.getRegionIterArgs())) {
              Value mappedArg = *newArgIter;
              if (useCompleteScalarAddress(forOp, slot, oldArg.getType()))
                mappedArg =
                    rebuildScalarPointer(mappedArg, oldArg.getType(), b, loc);
              mapping.map(oldArg, mappedArg);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : forOp.getBody()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     forOp));
          });
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      // Keep analyzable scalar pointers on the established relative-offset
      // path. If any boundary edge has no stable base-plus-offset form, switch
      // all scalar pointer edges atomically to complete i64 addresses so the
      // before/after regions cannot acquire a mixed pointer/integer contract.
      if (shouldPreserveScalarPointers(whileOp.getOperation(), rewriter,
                                       offsetMap))
        markOpaqueScalarPointerBoundary(whileOp.getOperation());
      SmallVector<Value> newInits = constructOperands(
          whileOp.getInits(), tempVar, mapping, rewriter, whileOp);
      newOp = rewriter.create<scf::WhileOp>(
          whileOp.getLoc(), constructTypes(whileOp->getResultTypes(), whileOp),
          newInits,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            IRMapping beforeMapping;
            auto newArgIter = args.begin();
            for (auto [slot, oldArg] :
                 llvm::enumerate(whileOp.getBeforeArguments())) {
              Value mappedArg = *newArgIter;
              if (useCompleteScalarAddress(whileOp, slot, oldArg.getType()))
                mappedArg =
                    rebuildScalarPointer(mappedArg, oldArg.getType(), b, loc);
              beforeMapping.map(oldArg, mappedArg);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : whileOp.getBeforeBody()->without_terminator()) {
              b.clone(bodyOp, beforeMapping);
            }
            auto conditionOp = whileOp.getConditionOp();
            b.create<scf::ConditionOp>(
                conditionOp.getLoc(),
                beforeMapping.lookup(conditionOp.getCondition()),
                constructOperands(conditionOp.getArgs(), tempVar, beforeMapping,
                                  b, whileOp));
          },
          [&](OpBuilder &b, Location loc, ValueRange args) {
            IRMapping afterMapping;
            auto newArgIter = args.begin();
            for (auto [slot, oldArg] :
                 llvm::enumerate(whileOp.getAfterArguments())) {
              Value mappedArg = *newArgIter;
              if (useCompleteScalarAddress(whileOp, slot, oldArg.getType()))
                mappedArg =
                    rebuildScalarPointer(mappedArg, oldArg.getType(), b, loc);
              afterMapping.map(oldArg, mappedArg);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : whileOp.getAfterBody()->without_terminator()) {
              b.clone(bodyOp, afterMapping);
            }
            auto yieldOp = whileOp.getYieldOp();
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, afterMapping, b,
                                                     whileOp));
          });
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op);
               ifOp && ifOp->getNumResults() > 0) {
      newOp = rewriter.create<scf::IfOp>(
          ifOp.getLoc(), ifOp.getCondition(),
          [&](OpBuilder &b, Location loc) {
            for (auto &bodyOp : ifOp.thenBlock()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = ifOp.thenYield();
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     /*loop=*/nullptr));
          },
          [&](OpBuilder &b, Location loc) {
            for (auto &bodyOp : ifOp.elseBlock()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = ifOp.elseYield();
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     /*loop=*/nullptr));
          });
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
      llvm_unreachable("Unsupported loop op");
    }
    if (newOp) {
      newOp->setAttrs(op->getAttrs());
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Converting\n" << *op << "\nto\n" << *newOp << "\n";
      });
      auto resIter = newOp->result_begin();
      for (auto [slot, res] : llvm::enumerate(op->getResults())) {
        Value replacement = *resIter;
        if (useCompleteScalarAddress(op, slot, res.getType())) {
          rewriter.setInsertionPointAfter(newOp);
          replacement = rebuildScalarPointer(replacement, res.getType(),
                                             rewriter, res.getLoc());
        }
        rewriter.replaceAllUsesWith(res, replacement);
        std::advance(resIter, std::max(getPtrTensorRank(res.getType()), 1));
      }
      rewriter.moveOpBefore(op, &retiredOps, retiredOps.end());
      op = newOp;
      convertTensorPtrPre(op, rewriter, offsetMap);
      for (auto &region : op->getRegions())
        region.walk<WalkOrder::PreOrder>(convertTensorPtr);
      convertTensorPtrPost(op, rewriter, offsetMap);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  };

  funcOp->walk<WalkOrder::PreOrder>(convertTensorPtr);
  // Drop all cached references before retiredOps destroys the old IR.
  offsetMap.clear();
}
