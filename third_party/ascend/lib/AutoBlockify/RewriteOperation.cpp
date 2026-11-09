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

#include "AutoBlockify/AutoBlockify.h"
#include "AutoBlockify/Utils.h"
#include "Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "auto-blockify-rewrite-operation"

using namespace mlir;
using namespace triton;

void PropagateUnrealizedCastDown::handleBlockifyLoop(
    scf::ForOp blockifyLoop, Operation *op, PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  for (auto opr : op->getOperands()) {
    auto uccOp = opr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!uccOp) {
      newOperands.push_back(opr);
      continue;
    }
    auto input = uccOp.getInputs()[0];
    auto tensorType = cast<RankedTensorType>(input.getType());
    Value newOperand;
    if (tensorType.getRank() > 1) {
      SmallVector<OpFoldResult> offsets(tensorType.getRank(),
                                        rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes(1, rewriter.getIndexAttr(1));
      SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                        rewriter.getIndexAttr(1));
      offsets[0] = blockifyLoop.getInductionVar();
      for (auto dim : llvm::drop_begin(tensorType.getShape()))
        sizes.push_back(rewriter.getIndexAttr(dim));
      newOperand = rewriter.create<tensor::ExtractSliceOp>(
          input.getLoc(), cast<RankedTensorType>(opr.getType()), input, offsets,
          sizes, strides);
    } else {
      newOperand = rewriter.create<tensor::ExtractOp>(
          input.getLoc(), input, ValueRange{blockifyLoop.getInductionVar()});
      if (isa<IndexType>(opr.getType())) {
        newOperand = rewriter.create<arith::IndexCastOp>(
            input.getLoc(), rewriter.getIndexType(), newOperand);
      }
    }
    newOperands.push_back(newOperand);
  }
  rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newOperands); });
}

void PropagateUnrealizedCastDown::rewriteGeneraleOp(
    UnrealizedConversionCastOp op, Operation *generalOp,
    PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto res = op->getResult(0);
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<Value> newOperands;
  SmallVector<Value> newResults;
  SmallVector<Type> newResultTypes;

  for (auto operand : generalOp->getOperands())
    newOperands.push_back(rewriteValue(operand, op, rewriter));
  for (auto resType : generalOp->getResultTypes()) {
    newResultTypes.push_back(getExpandedType(resType, op));
  }
  auto *newOp =
      rewriter.create(generalOp->getLoc(), generalOp->getName().getIdentifier(),
                      newOperands, newResultTypes, generalOp->getAttrs());
  replaceValue(newOp, generalOp, mask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteSplat(
    UnrealizedConversionCastOp op, triton::SplatOp splatOp,
    PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto resType = cast<RankedTensorType>(splatOp.getResult().getType());
  auto curShape =
      llvm::to_vector(cast<RankedTensorType>(input.getType()).getShape());
  auto splatedShape = resType.getShape();
  for (auto dim : splatedShape) {
    input = rewriter.create<triton::ExpandDimsOp>(input.getLoc(), input,
                                                  curShape.size());
    curShape.push_back(dim);
    input = rewriter.create<triton::BroadcastOp>(
        input.getLoc(),
        RankedTensorType::get(curShape, getElementTypeOrSelf(input)), input);
  }
  replaceValue(input.getDefiningOp(), splatOp, mask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteExpandDims(
    UnrealizedConversionCastOp op, triton::ExpandDimsOp expandDimsOp,
    PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto newOp = rewriter.create<triton::ExpandDimsOp>(
      expandDimsOp.getLoc(), input, expandDimsOp.getAxis() + 1);
  for (auto attr : expandDimsOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, expandDimsOp, mask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteReduce(
    UnrealizedConversionCastOp op, triton::ReduceOp reduceOp,
    PatternRewriter &rewriter) const {
  auto mask = op.getInputs()[1];
  auto srcs = llvm::map_to_vector(reduceOp.getSrcs(), [&](Value src) {
    return rewriteValue(src, op, rewriter);
  });
  auto newOp = rewriter.create<triton::ReduceOp>(reduceOp.getLoc(), srcs,
                                                 reduceOp.getAxis() + 1);
  auto &newCombineOp = newOp.getCombineOp();
  rewriter.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp,
                             newCombineOp.end());
  for (auto attr : reduceOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, reduceOp, mask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteScan(UnrealizedConversionCastOp op,
                                              triton::ScanOp scanOp,
                                              PatternRewriter &rewriter) const {
  auto mask = op.getInputs()[1];
  auto srcs = llvm::map_to_vector(scanOp.getSrcs(), [&](Value src) {
    return rewriteValue(src, op, rewriter);
  });
  auto newOp = rewriter.create<triton::ScanOp>(
      scanOp.getLoc(), srcs, scanOp.getAxis() + 1, scanOp.getReverse());
  auto &newCombineOp = newOp.getCombineOp();
  rewriter.cloneRegionBefore(scanOp.getCombineOp(), newCombineOp,
                             newCombineOp.end());
  for (auto attr : scanOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, scanOp, mask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteLoad(UnrealizedConversionCastOp op,
                                              triton::LoadOp loadOp,
                                              PatternRewriter &rewriter) const {
  auto uccMask = op.getInputs()[1];
  auto ptr = rewriteValue(loadOp.getPtr(), op, rewriter);
  auto other = rewriteValue(loadOp.getOther(), op, rewriter);
  auto mask = rewriteValue(loadOp.getMask(), op, rewriter);
  auto res = loadOp.getResult();
  auto resType = getExpandedType(res.getType(), op);
  if (!other) {
    other = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(),
        DenseElementsAttr::get(
            resType, rewriter.getZeroAttr(getElementTypeOrSelf(res))));
  }
  mask = createMask(mask, uccMask, resType.getShape(), rewriter);
  auto boundaryCheck = llvm::map_to_vector(loadOp.getBoundaryCheck(),
                                           [](int32_t idx) { return idx + 1; });
  auto newOp = rewriter.create<triton::LoadOp>(
      loadOp.getLoc(), ptr, mask, other, boundaryCheck, loadOp.getPadding(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
  for (auto attr : loadOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, loadOp, uccMask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteStore(
    UnrealizedConversionCastOp op, triton::StoreOp storeOp,
    PatternRewriter &rewriter) const {
  auto uccMask = op.getInputs()[1];
  auto ptr = rewriteValue(storeOp.getPtr(), op, rewriter);
  auto value = rewriteValue(storeOp.getValue(), op, rewriter);
  auto mask = rewriteValue(storeOp.getMask(), op, rewriter);
  auto ptrShape = cast<RankedTensorType>(ptr.getType()).getShape();
  mask = createMask(mask, uccMask, ptrShape, rewriter);
  auto boundaryCheck = llvm::map_to_vector(storeOp.getBoundaryCheck(),
                                           [](int32_t idx) { return idx + 1; });
  auto newOp = rewriter.create<triton::StoreOp>(
      storeOp.getLoc(), ptr, value, mask, boundaryCheck, storeOp.getCache(),
      storeOp.getEvict());
  for (auto attr : storeOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  rewriter.replaceOp(storeOp, newOp);
}

void PropagateUnrealizedCastDown::rewriteAtomicRMW(
    UnrealizedConversionCastOp op, triton::AtomicRMWOp atomicRMWOp,
    PatternRewriter &rewriter) const {
  auto uccMask = op.getInputs()[1];
  auto ptr = rewriteValue(atomicRMWOp.getPtr(), op, rewriter);
  auto val = rewriteValue(atomicRMWOp.getVal(), op, rewriter);
  auto mask = rewriteValue(atomicRMWOp.getMask(), op, rewriter);
  auto resType = getExpandedType(atomicRMWOp.getResult().getType(), op);
  mask = createMask(mask, uccMask, resType.getShape(), rewriter);
  auto newOp = rewriter.create<triton::AtomicRMWOp>(
      atomicRMWOp.getLoc(), resType, atomicRMWOp.getAtomicRmwOp(), ptr, val,
      mask, atomicRMWOp.getSem(), atomicRMWOp.getScope());
  for (auto attr : atomicRMWOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, atomicRMWOp, uccMask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteAssert(
    UnrealizedConversionCastOp op, triton::AssertOp assertOp,
    PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
  auto conditionType = cast<RankedTensorType>(mask.getType());
  auto oneAttr = rewriter.getIntegerAttr(getElementTypeOrSelf(mask), 1);
  auto one = rewriter.create<arith::ConstantOp>(
      mask.getLoc(), DenseElementsAttr::get(conditionType, oneAttr));
  Value condition = rewriter.create<arith::XOrIOp>(input.getLoc(), mask, one);
  condition = createMask(nullptr, condition, inputShape, rewriter);
  condition =
      rewriter.create<arith::OrIOp>(condition.getLoc(), condition, input);
  auto newOp = rewriter.create<triton::AssertOp>(assertOp.getLoc(), condition,
                                                 assertOp.getMessage());
  for (auto attr : assertOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  rewriter.replaceOp(assertOp, newOp);
}

void PropagateUnrealizedCastDown::rewriteExtractSlice(
    UnrealizedConversionCastOp op, tensor::ExtractSliceOp extractSliceOp,
    PatternRewriter &rewriter) const {
  auto mask = op.getInputs()[1];
  auto src = rewriteValue(extractSliceOp.getSource(), op, rewriter);
  auto offsets = llvm::to_vector(extractSliceOp.getMixedOffsets());
  auto sizes = llvm::to_vector(extractSliceOp.getMixedSizes());
  auto strides = llvm::to_vector(extractSliceOp.getMixedStrides());
  auto srcType = cast<RankedTensorType>(src.getType());
  offsets.insert(offsets.begin(), rewriter.getIndexAttr(0));
  sizes.insert(sizes.begin(), rewriter.getIndexAttr(srcType.getShape()[0]));
  strides.insert(strides.begin(), rewriter.getIndexAttr(1));
  auto newOp = rewriter.create<tensor::ExtractSliceOp>(
      extractSliceOp.getLoc(), src, offsets, sizes, strides);
  auto newMask = rewriter.create<tensor::ExtractSliceOp>(
      mask.getLoc(), mask, offsets, sizes, strides);
  for (auto attr : extractSliceOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, extractSliceOp, newMask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteInsertSlice(
    UnrealizedConversionCastOp op, tensor::InsertSliceOp insertSliceOp,
    PatternRewriter &rewriter) const {
  auto mask = op.getInputs()[1];
  auto src = rewriteValue(insertSliceOp.getSource(), op, rewriter);
  auto dst = rewriteValue(insertSliceOp.getDest(), op, rewriter);
  auto offsets = llvm::to_vector(insertSliceOp.getMixedOffsets());
  auto sizes = llvm::to_vector(insertSliceOp.getMixedSizes());
  auto strides = llvm::to_vector(insertSliceOp.getMixedStrides());
  auto srcType = cast<RankedTensorType>(src.getType());
  offsets.insert(offsets.begin(), rewriter.getIndexAttr(0));
  sizes.insert(sizes.begin(), rewriter.getIndexAttr(srcType.getShape()[0]));
  strides.insert(strides.begin(), rewriter.getIndexAttr(1));
  auto newOp = rewriter.create<tensor::InsertSliceOp>(
      insertSliceOp.getLoc(), src, dst, offsets, sizes, strides);
  for (auto attr : insertSliceOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  replaceValue(newOp, insertSliceOp, mask, rewriter);
}

void PropagateUnrealizedCastDown::rewriteWhile(
    UnrealizedConversionCastOp op, scf::WhileOp whileOp,
    PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto res = op->getResult(0);
  SmallVector<int64_t> indices;
  SmallVector<Value> newInits;
  IRMapping mapping;
  for (auto [idx, init] : llvm::enumerate(whileOp.getInits())) {
    if (init == res) {
      indices.push_back(idx);
      newInits.push_back(input);
    } else {
      newInits.push_back(init);
    }
  }
  auto newOp = rewriter.create<scf::WhileOp>(
      whileOp.getLoc(), whileOp->getResultTypes(), newInits,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        mapRegionIterArg(mapping, whileOp.getBeforeArguments(), args, indices,
                         mask, b);
        for (auto &bodyOp : *whileOp.getBeforeBody())
          b.clone(bodyOp, mapping);
      },
      [&](OpBuilder &b, Location loc, ValueRange args) {
        mapRegionIterArg(mapping, whileOp.getAfterArguments(), args, {}, mask,
                         b);
        for (auto &bodyOp : whileOp.getAfterBody()->without_terminator())
          b.clone(bodyOp, mapping);
        auto yieldOp =
            cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator());
        mapYieldedValue(mapping, yieldOp, indices, op, b);
      });
  for (auto attr : whileOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  rewriter.replaceOp(whileOp, newOp);
}

void PropagateUnrealizedCastDown::rewriteLoop(UnrealizedConversionCastOp op,
                                              LoopLikeOpInterface loopOp,
                                              PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto res = op->getResult(0);
  SmallVector<int64_t> indices;
  SmallVector<Value> newInits;
  IRMapping mapping;
  for (auto [idx, init] : llvm::enumerate(loopOp.getInits())) {
    if (init == res) {
      indices.push_back(idx);
      newInits.push_back(input);
    } else {
      newInits.push_back(init);
    }
  }
  LoopLikeOpInterface newOp;
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation())) {
    newOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          mapping.map(forOp.getInductionVar(), iv);
          mapRegionIterArg(mapping, forOp.getRegionIterArgs(), args, indices,
                           mask, b);
          for (auto &bodyOp : forOp.getBody()->without_terminator())
            b.clone(bodyOp, mapping);
          auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          mapYieldedValue(mapping, yieldOp, indices, op, b);
        });
    for (auto attr : forOp->getAttrs()) {
      if (!newOp->hasAttr(attr.getName()))
        newOp->setAttr(attr.getName(), attr.getValue());
    }
  } else {
    llvm_unreachable("Unhandled loopOp");
  }
  replaceValue(newOp, loopOp, mask, rewriter, indices);
}

void PropagateUnrealizedCastDown::rewriteIf(UnrealizedConversionCastOp &op,
                                            scf::IfOp ifOp,
                                            ArrayRef<int64_t> indices,
                                            PatternRewriter &rewriter) const {
  IRMapping mapping;
  auto mask = op.getInputs()[1];
  auto thenBlockBuilder = [&](OpBuilder &b, Location loc) {
    for (auto &bodyOp : *ifOp.thenBlock())
      b.clone(bodyOp, mapping);
  };
  auto elseBlockBuilder = [&](OpBuilder &b, Location loc) {
    for (auto &bodyOp : *ifOp.elseBlock())
      b.clone(bodyOp, mapping);
  };
  scf::IfOp newOp;
  if (ifOp.elseBlock()) {
    newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), ifOp.getCondition(),
                                       thenBlockBuilder, elseBlockBuilder);
  } else {
    newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), ifOp.getCondition(),
                                       thenBlockBuilder, nullptr);
  }
  for (auto attr : ifOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  if (mapping.contains(op))
    op = cast<UnrealizedConversionCastOp>(mapping.lookup(op));
  replaceValue(newOp, ifOp, mask, rewriter, indices);
}

void PropagateUnrealizedCastDown::rewriteYield(
    UnrealizedConversionCastOp &op, scf::YieldOp yieldOp,
    PatternRewriter &rewriter) const {
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto res = op->getResult(0);
  SmallVector<int64_t> indices;
  auto newOperands = llvm::to_vector(yieldOp.getOperands());
  for (auto [idx, opr] : llvm::enumerate(newOperands)) {
    if (opr == res)
      indices.push_back(idx);
  }
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp())) {
    auto uccOp = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), res.getType(), ValueRange({input}));
    for (auto curIdx : indices)
      newOperands[curIdx] = uccOp->getResult(0);
    auto newOp = rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newOperands);
    for (auto attr : yieldOp->getAttrs()) {
      if (!newOp->hasAttr(attr.getName()))
        newOp->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(yieldOp, newOp);
    rewriter.setInsertionPoint(loopOp);
    for (auto curIdx : indices) {
      auto &initArg = loopOp.getInitsMutable()[curIdx];
      auto initVal = initArg.get();
      uccOp = rewriter.create<UnrealizedConversionCastOp>(
          initVal.getLoc(), input.getType(), ValueRange({initVal}));
      uccOp = rewriter.create<UnrealizedConversionCastOp>(
          initVal.getLoc(), initVal.getType(),
          ValueRange({uccOp->getResult(0), mask}));
      rewriter.modifyOpInPlace(loopOp,
                               [&]() { initArg.set(uccOp->getResult(0)); });
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
    for (auto curIdx : indices)
      newOperands[curIdx] = input;
    auto newOp = rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newOperands);
    for (auto attr : yieldOp->getAttrs()) {
      if (!newOp->hasAttr(attr.getName()))
        newOp->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(yieldOp, newOp);
    yieldOp = ifOp.thenYield() == yieldOp ? ifOp.elseYield() : ifOp.thenYield();
    if (yieldOp) {
      rewriter.setInsertionPoint(yieldOp);
      newOperands = llvm::to_vector(yieldOp.getOperands());
      for (auto curIdx : indices) {
        auto uccOp = rewriter.create<UnrealizedConversionCastOp>(
            op.getLoc(), input.getType(), ValueRange({newOperands[curIdx]}));
        newOperands[curIdx] = uccOp->getResult(0);
      }
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newOperands);
    }
    rewriter.setInsertionPoint(ifOp);
    rewriteIf(op, ifOp, indices, rewriter);
  }
}

void PropagateUnrealizedCastDown::rewriteCondition(
    UnrealizedConversionCastOp op, scf::ConditionOp conditionOp,
    PatternRewriter &rewriter) const {
  auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  auto input = op.getInputs()[0];
  auto mask = op.getInputs()[1];
  auto res = op->getResult(0);
  int64_t curIdx = -1;
  auto args = llvm::to_vector(conditionOp.getArgs());
  for (auto [idx, opr] : llvm::enumerate(args)) {
    if (opr == res)
      curIdx = idx;
  }
  args[curIdx] = input;
  auto newOp = rewriter.create<scf::ConditionOp>(
      conditionOp.getLoc(), conditionOp.getCondition(), args);
  for (auto attr : conditionOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  rewriter.replaceOp(conditionOp, newOp);

  res = whileOp->getResult(curIdx);
  auto oldResType = res.getType();
  auto newResType = getExpandedType(oldResType, op);
  rewriter.modifyOpInPlace(whileOp, [&]() { res.setType(newResType); });
  rewriter.setInsertionPointAfter(whileOp);
  auto newUccOp = rewriter.create<UnrealizedConversionCastOp>(
      res.getLoc(), oldResType, ValueRange({res, mask}));
  rewriter.replaceAllUsesExcept(res, newUccOp->getResult(0), newUccOp);
  auto arg = whileOp.getAfterArguments()[curIdx];
  auto oldArgType = arg.getType();
  auto newArgType = getExpandedType(oldArgType, op);
  rewriter.modifyOpInPlace(whileOp, [&]() { arg.setType(newArgType); });
  rewriter.setInsertionPointToStart(whileOp.getAfterBody());
  newUccOp = rewriter.create<UnrealizedConversionCastOp>(
      arg.getLoc(), oldArgType, ValueRange({arg, mask}));
  rewriter.replaceAllUsesExcept(arg, newUccOp->getResult(0), newUccOp);
}
