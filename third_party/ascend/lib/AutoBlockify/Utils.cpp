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

#include "AutoBlockify/Utils.h"
#include "Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "auto-blockify-utils"

using namespace mlir;
using namespace triton;

RankedTensorType getExpandedType(Type type, UnrealizedConversionCastOp op) {
  auto target = op.getInputs()[0];
  auto targetType = cast<RankedTensorType>(target.getType());
  SmallVector<int64_t> targetShape{targetType.getShape()[0]};
  if (auto valueType = dyn_cast<RankedTensorType>(type)) {
    targetShape.append(valueType.getShape().begin(),
                       valueType.getShape().end());
  }
  return RankedTensorType::get(targetShape, getElementTypeOrSelf(type));
}

Value rewriteValue(Value value, UnrealizedConversionCastOp op,
                   OpBuilder &builder) {
  if (value == nullptr)
    return nullptr;
  if (value == op->getResult(0))
    return op.getInputs()[0];
  return builder
      .create<UnrealizedConversionCastOp>(
          value.getLoc(), getExpandedType(value.getType(), op), value)
      ->getResult(0);
}

void replaceValue(Operation *newOp, Operation *oldOp, Value newMask,
                  RewriterBase &rewriter, ArrayRef<int64_t> replaceIndices) {
  int64_t idx = 0;
  for (auto [res, oldRes] :
       llvm::zip_equal(newOp->getResults(), oldOp->getResults())) {
    if (replaceIndices.empty() ||
        llvm::find(replaceIndices, idx) != replaceIndices.end()) {
      auto resType = res.getType();
      auto newUccOp = rewriter.create<UnrealizedConversionCastOp>(
          newOp->getLoc(), oldRes.getType(), ValueRange({res, newMask}));
      rewriter.replaceAllUsesExcept(oldRes, newUccOp->getResult(0), newUccOp);
    } else {
      rewriter.replaceAllUsesWith(oldRes, res);
    }
    idx++;
  }
  rewriter.eraseOp(oldOp);
}

Value createMask(Value mask, Value uccMask, ArrayRef<int64_t> targetShape,
                 RewriterBase &rewriter) {
  SmallVector<int64_t> curShape{targetShape[0]};
  for (auto [idx, dim] : llvm::drop_begin(llvm::enumerate(targetShape))) {
    curShape.push_back(dim);
    uccMask =
        rewriter.create<triton::ExpandDimsOp>(uccMask.getLoc(), uccMask, idx);
    uccMask = rewriter.create<triton::BroadcastOp>(
        uccMask.getLoc(),
        RankedTensorType::get(curShape, getElementTypeOrSelf(uccMask)),
        uccMask);
  }
  if (mask) {
    mask = rewriter.create<arith::AndIOp>(mask.getLoc(), mask, uccMask);
  } else {
    mask = uccMask;
  }
  return mask;
}

void mapRegionIterArg(IRMapping &mapping, ValueRange oldArgs,
                      ValueRange newArgs, ArrayRef<int64_t> indices, Value mask,
                      OpBuilder &builder) {
  auto newArgIter = newArgs.begin();
  for (auto [idx, oldArg] : llvm::enumerate(oldArgs)) {
    if (llvm::find(indices, idx) != indices.end()) {
      auto newUccOp = builder.create<UnrealizedConversionCastOp>(
          oldArg.getLoc(), oldArg.getType(), ValueRange({*newArgIter, mask}));
      mapping.map(oldArg, newUccOp->getResult(0));
    } else {
      mapping.map(oldArg, *newArgIter);
    }
    ++newArgIter;
  }
}

void mapYieldedValue(IRMapping &mapping, scf::YieldOp yieldOp,
                     ArrayRef<int64_t> indices, UnrealizedConversionCastOp op,
                     OpBuilder &builder) {
  SmallVector<Value> newOperands;
  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    operand = mapping.lookup(operand);
    if (llvm::find(indices, idx) != indices.end())
      newOperands.push_back(rewriteValue(operand, op, builder));
    else
      newOperands.push_back(operand);
  }
  builder.create<scf::YieldOp>(yieldOp.getLoc(), newOperands);
}

Operation *createBlockifyLoop(Operation *targetOp,
                              UnrealizedConversionCastOp op,
                              Value logicalBlockId, Value logicalBlockNum,
                              int autoBlockifySize, RewriterBase &rewriter) {
  auto loc = targetOp->getLoc();
  rewriter.setInsertionPoint(targetOp);
  auto initVal =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  auto stepVal =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto blockifySizeVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexAttr(autoBlockifySize));
  Value upperBound =
      rewriter.create<arith::SubIOp>(loc, logicalBlockNum, logicalBlockId);
  auto i32Zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  upperBound = rewriter.create<arith::MaxSIOp>(loc, upperBound, i32Zero);
  upperBound = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   upperBound);
  upperBound =
      rewriter.create<arith::MinSIOp>(loc, upperBound, blockifySizeVal);
  SmallVector<Value> inits;
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(targetOp)) {
    inits = llvm::map_to_vector(loopOp.getInits(),
                                [&rewriter, &op](Value v) -> Value {
                                  return rewriteValue(v, op, rewriter);
                                });
  } else {
    auto resultTypes =
        llvm::map_to_vector(targetOp->getResultTypes(), [&op](Type type) {
          return getExpandedType(type, op);
        });
    inits =
        llvm::map_to_vector(resultTypes, [&rewriter, &loc](Type type) -> Value {
          auto tensorType = cast<RankedTensorType>(type);
          return rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
        });
  }
  auto mask = op.getInputs()[1];
  Operation *newOp;
  auto blockifyLoop = rewriter.create<scf::ForOp>(
      loc, initVal, upperBound, stepVal, inits,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        newOp = b.clone(*targetOp);

        SmallVector<Value> newResults;
        for (auto [arg, res] : llvm::zip_equal(args, newOp->getResults())) {
          auto tensorType = cast<RankedTensorType>(arg.getType());
          auto rank = tensorType.getRank();
          Value newRes;
          if (rank > 1) {
            SmallVector<OpFoldResult> offsets(tensorType.getRank(),
                                              b.getIndexAttr(0));
            SmallVector<OpFoldResult> sizes(1, b.getIndexAttr(1));
            SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                              b.getIndexAttr(1));
            offsets[0] = iv;
            for (auto dim : llvm::drop_begin(tensorType.getShape()))
              sizes.push_back(b.getIndexAttr(dim));
            newRes = b.create<tensor::InsertSliceOp>(loc, res, arg, offsets,
                                                     sizes, strides);
          } else {
            newRes = b.create<tensor::InsertOp>(loc, res, arg, ValueRange{iv});
          }
          newResults.push_back(newRes);
        }
        b.create<scf::YieldOp>(loc, newResults);
      });

  replaceValue(blockifyLoop, targetOp, mask, rewriter);
  blockifyLoop->setAttr(autoBlockifyLoopAttr, rewriter.getUnitAttr());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After creating blockify loop:\n" << blockifyLoop << "\n";
  });
  return newOp;
}

std::optional<scf::ForOp> getBlockifyLoop(Operation *op) {
  while (auto forOp = op->getParentOfType<scf::ForOp>()) {
    if (forOp->hasAttr(autoBlockifyLoopAttr))
      return forOp;
    op = forOp;
  }
  return std::nullopt;
}