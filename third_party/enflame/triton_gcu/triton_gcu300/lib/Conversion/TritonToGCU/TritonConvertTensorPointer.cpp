/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
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
#include <memory>
#include <stack>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/Debug.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define DEBUG_TYPE "triton-convert-tensor-pointer"
namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORPOINTERPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

/// An additional struct to record the meta information of operations
/// with tensor pointers
struct RewritedInfo {
private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tensorShape;
  Attribute encoding;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;

public:
  RewritedInfo() = default;

  RewritedInfo(const RewritedInfo &other) = default;
  RewritedInfo &operator=(const RewritedInfo &other) = default;

  RewritedInfo(Value base, const SmallVector<Value> &shape,
               const SmallVector<Value> &strides,
               const SmallVector<Value> &offsets,
               const ArrayRef<int64_t> &tensorShape, const Attribute &encoding)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tensorShape(tensorShape), encoding(encoding) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tensorShape.size());
  }

  unsigned int length() const { return shape.size(); }

  Value getOffset(unsigned i) { return offsets[i]; }

  SmallVector<Value> getOffsets() { return offsets; }

  void setOffset(unsigned i, Value newOffset) {
    offsets[i] = newOffset;
    cachedOffsetWithRange.clear();
  }

  void setOffsets(const SmallVector<Value> &newOffsets) {
    offsets = newOffsets;
    cachedOffsetWithRange.clear();
  }

  Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                   unsigned i) {
    if (cachedOffsetWithRange.count(i))
      return cachedOffsetWithRange[i];

    auto srcEncoding = cast<triton::gpu::DistributedEncodingTrait>(encoding);
    for (int j = tensorShape.size() - 1; j >= 0; j--) {
      if (static_cast<unsigned>(j) == i)
        continue;
      srcEncoding = triton::gpu::SliceEncodingAttr::get(
          builder.getContext(), j,
          cast<triton::gpu::DistributedEncodingTrait>(srcEncoding));
    }

    auto indexI32RowType = RankedTensorType::get(
        {tensorShape[i]}, builder.getI32Type(), srcEncoding);

    Value splatOffset =
        builder.create<triton::SplatOp>(loc, indexI32RowType, offsets[i]);
    Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                      tensorShape[i]);
    // Expand dimensions
    Value expandedResult =
        builder.create<arith::AddIOp>(loc, splatOffset, range);

    for (unsigned j = 0; j < tensorShape.size(); ++j) {
      if (j == i)
        continue;

      expandedResult =
          builder.create<triton::ExpandDimsOp>(loc, expandedResult, j);
    }

    return cachedOffsetWithRange[i] = expandedResult;
  }

  Value generatePtr(OpBuilder &builder, const Location &loc) {
    assert(tensorShape.size() == offsets.size() &&
           tensorShape.size() == strides.size());

    auto indexTensorType =
        RankedTensorType::get(tensorShape, builder.getI32Type(), encoding);
    auto ptrType = cast<triton::PointerType>(base.getType());
    auto ptrTensorType = RankedTensorType::get(tensorShape, ptrType, encoding);

    // Generate offsets per dimension
    Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, base);
    for (unsigned i = 0; i < tensorShape.size(); ++i) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // We must splat strides into the expanded shape not a row for retaining
      // the divisibility information given by strides
      Value splatStride = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), strides[i]);
      Value offsetWithStride =
          builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
      Value broadcasted = builder.create<triton::BroadcastOp>(
          loc, indexTensorType, offsetWithStride);

      // Add to the pointer
      ptr = builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr,
                                             broadcasted);
    }

    return ptr;
  }

  Value generateMask(OpBuilder &builder, const Location &loc,
                     const std::optional<ArrayRef<int32_t>> &boundaryCheck) {
    if (!boundaryCheck.has_value())
      return {};

    // Generate mask per dimension
    auto maskTensorType =
        RankedTensorType::get(tensorShape, builder.getI1Type(), encoding);
    Value mask;
    for (auto i : boundaryCheck.value()) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // Compare with lower bound
      Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
          loc, 0, builder.getI32Type());
      Value splatLowerBound = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), lowerBound);
      Value cmpLower = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

      // Compare with upper bound
      Value splatUpperBound = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), shape[i]);
      Value cmpUpper = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, offsetWithRange, splatUpperBound);

      // And and broadcast
      Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
      Value broadcasted =
          builder.create<triton::BroadcastOp>(loc, maskTensorType, andResult);

      // And up all results
      if (!mask) {
        mask = broadcasted;
      } else {
        mask = builder.create<arith::AndIOp>(loc, mask, broadcasted);
      }
    }

    return mask;
  }

  Value generateOther(OpBuilder &builder, const Location &loc,
                      const std::optional<triton::PaddingOption> &padding) {
    if (!padding.has_value())
      return Value();

    // Create element attribute
    auto elementType =
        cast<triton::PointerType>(base.getType()).getPointeeType();
    auto otherTensorType =
        RankedTensorType::get(tensorShape, elementType, encoding);

    // Set zero padding value
    TypedAttr attr =
        elementType.isIntOrIndex()
            ? cast<TypedAttr>(builder.getIntegerAttr(elementType, 0))
            : cast<TypedAttr>(builder.getFloatAttr(elementType, 0));

    // Float NaN padding case
    if (padding.value() == triton::PaddingOption::PAD_NAN) {
      assert(!elementType.isIntOrIndex());
      auto apNaN = llvm::APFloat::getNaN(
          cast<FloatAttr>(attr).getValue().getSemantics());
      attr = builder.getFloatAttr(elementType, apNaN);
    }

    // Create tensor
    Value constant = builder.create<arith::ConstantOp>(loc, attr);
    return builder.create<triton::SplatOp>(loc, otherTensorType, constant);
  }
};

static bool needRewrite(Operation *op) {
  return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                     [](Value operand) {
                       return triton::isTensorPointerType(operand.getType());
                     });
}

static SmallVector<Value>
generateNewOperands(const SmallVector<Value> &oldOperands, unsigned index,
                    const SmallVector<Value> &newValues) {
  assert(index < oldOperands.size());
  SmallVector<Value> newOperands;
  for (unsigned i = 0; i < index; ++i)
    newOperands.push_back(oldOperands[i]);
  for (auto value : newValues)
    newOperands.push_back(value);
  for (auto i = index + 1; i < oldOperands.size(); ++i)
    newOperands.push_back(oldOperands[i]);
  return newOperands;
}

struct ConvertTensorPointerPass
    : public mlir::impl::ConvertTensorPointerPassBase<
          ConvertTensorPointerPass> {
  using Base::Base;
  void runOnOperation() override;

  Operation *rewriteMakeTensorPtrOp(OpBuilder &builder,
                                    triton::MakeTensorPtrOp op,
                                    std::stack<Operation *> &eraser);
  Operation *rewriteAdvanceOp(OpBuilder &builder, triton::AdvanceOp op,
                              std::stack<Operation *> &eraser);
  Operation *rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                std::stack<Operation *> &eraser);
  Operation *rewriteIfOp(OpBuilder &builder, scf::IfOp op,
                         std::stack<Operation *> &eraser);
  Operation *rewriteForOp(OpBuilder &builder, scf::ForOp op,
                          std::stack<Operation *> &eraser);
  Operation *rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                            std::stack<Operation *> &eraser);
  Operation *rewriteOp(Operation *op, std::stack<Operation *> &eraser);
  void visitOperation(Operation *op, std::stack<Operation *> &eraser);

private:
  DenseMap<Value, RewritedInfo> rewritedInfo;
};

} // namespace

Operation *ConvertTensorPointerPass::rewriteMakeTensorPtrOp(
    OpBuilder &builder, triton::MakeTensorPtrOp op,
    std::stack<Operation *> &eraser) {
  // Save info for later use
  auto ptrType = cast<triton::PointerType>(op.getType());
  auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

  // Cast I64 Shape,Strides into I32
  SmallVector<Value> i32Shape, i32Strides;
  for (auto dim : op.getShape()) {
    auto i32Dim =
        builder.create<arith::TruncIOp>(op.getLoc(), builder.getI32Type(), dim);
    i32Shape.push_back(i32Dim);
  }

  for (auto stride : op.getStrides()) {
    auto i32Stride = builder.create<arith::TruncIOp>(
        op.getLoc(), builder.getI32Type(), stride);
    i32Strides.push_back(i32Stride);
  }

  // Save information
  rewritedInfo[op.getResult()] =
      RewritedInfo(op.getBase(), i32Shape, i32Strides, op.getOffsets(),
                   tensorType.getShape(), tensorType.getEncoding());

  // Erase the original operation
  eraser.push(op);
  return nullptr;
}

Operation *ConvertTensorPointerPass::rewriteAdvanceOp(
    OpBuilder &builder, triton::AdvanceOp op, std::stack<Operation *> &eraser) {
  // Get info from previous results
  assert(rewritedInfo.count(op.getPtr()));
  auto info = rewritedInfo[op.getPtr()];

  // Calculate new offsets
  assert(info.length() == op.getOffsets().size());
  SmallVector<Value> newOffsets;
  for (unsigned i = 0; i < info.length(); ++i) {
    // Value i64Offset = builder.create<arith::ExtSIOp>(
    //     op.getLoc(), builder.getI64Type(), op.getOffsets()[i]);
    Value newOffset = builder.create<arith::AddIOp>(
        op.getLoc(), info.getOffset(i), /*i64Offset*/ op.getOffsets()[i]);
    newOffsets.push_back(newOffset);
  }

  // Save info for later use
  info.setOffsets(newOffsets);
  rewritedInfo[op.getResult()] = info;

  // Erase the original operation
  eraser.push(op);
  return nullptr;
}

Operation *
ConvertTensorPointerPass::rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                             std::stack<Operation *> &eraser) {
  assert(isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op));

  // We only have to rewrite load/stores with tensor pointers
  auto ptr = op->getOperand(0);
  if (!triton::isTensorPointerType(ptr.getType()))
    return nullptr;

  // Get info from previous results
  assert(rewritedInfo.count(ptr));
  auto info = rewritedInfo[ptr];

  // Load/store with tensor pointers implicitly will check the bound while
  // accessing memory, so we should set `mask` and `other` (according to the
  // padding). Also note that load with tensor pointers do not have `mask` and
  // `other` while building IR from Python AST
  std::optional<ArrayRef<int>> boundaryCheck;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    assert(!loadOp.getMask() && !loadOp.getOther());
    boundaryCheck = loadOp.getBoundaryCheck();
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    assert(!storeOp.getMask());
    boundaryCheck = storeOp.getBoundaryCheck();
  }

  // Generate new `ptr`, `mask` and `other`
  auto newPtr = info.generatePtr(builder, op->getLoc());
  auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
  Value newOther;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op))
    newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

  // Create a new operation
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    auto newResult = builder.create<triton::LoadOp>(
        loadOp.getLoc(), newPtr, newMask, newOther, loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    op->getResult(0).replaceAllUsesWith(newResult);
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    builder.create<triton::StoreOp>(storeOp.getLoc(), newPtr,
                                    storeOp.getValue(), newMask,
                                    storeOp.getCache(), storeOp.getEvict());
  }

  // Erase the original operation
  eraser.push(op);
  return nullptr;
}

Operation *
ConvertTensorPointerPass::rewriteIfOp(OpBuilder &builder, scf::IfOp op,
                                      std::stack<Operation *> &eraser) {
  auto thenYieldOp = op.thenYield();
  assert(op.getNumResults() == thenYieldOp.getNumOperands());
  SmallVector<Value> results = thenYieldOp.getOperands();

  // get new result types
  SmallVector<Type> newRetTypes;
  bool needRewrite = false;
  for (unsigned i = 0; i < results.size(); ++i) {
    if (!triton::isTensorPointerType(results[i].getType())) {
      newRetTypes.push_back(results[i].getType());
      continue;
    }
    needRewrite = true;
    auto makeTensorPtrOp = getMakeTensorPtrOp(results[i]);
    assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
    auto info = rewritedInfo[makeTensorPtrOp.getResult()];
    for (unsigned j = 0; j < info.length(); ++j) {
      newRetTypes.push_back(builder.getI64Type());
    }
  }
  if (!needRewrite)
    return op;
  // create and clone new IfOp
  bool hasElse = !op.getElseRegion().empty();
  scf::IfOp newOp = builder.create<scf::IfOp>(op.getLoc(), newRetTypes,
                                              op.getCondition(), hasElse);
  IRMapping mapping;
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    mapping.map(op->getOperand(i), newOp->getOperand(i));
  }
  auto rematerialize = [&](Block *block) {
    for (Operation &opInIf : block->getOperations()) {
      builder.clone(opInIf, mapping);
    }
  };
  builder.setInsertionPointToStart(newOp.thenBlock());
  rematerialize(op.thenBlock());
  if (hasElse) {
    builder.setInsertionPointToStart(newOp.elseBlock());
    rematerialize(op.elseBlock());
  }

  // update rewritedInfo
  unsigned oldResIdx = 0, newResIdx = 0;
  while (oldResIdx < results.size()) {
    if (!triton::isTensorPointerType(results[oldResIdx].getType())) {
      oldResIdx++;
      newResIdx++;
    } else {
      auto makeTensorPtrOp = getMakeTensorPtrOp(results[oldResIdx]);
      assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
      auto info = rewritedInfo[makeTensorPtrOp.getResult()];
      for (unsigned j = 0; j < info.length(); ++j) {
        info.setOffset(j, newOp->getResult(newResIdx++));
      }
      rewritedInfo[op.getResult(oldResIdx)] = info;
      oldResIdx++;
    }
  }

  eraser.push(op);
  return newOp;
}

Operation *
ConvertTensorPointerPass::rewriteForOp(OpBuilder &builder, scf::ForOp op,
                                       std::stack<Operation *> &eraser) {
  SmallVector<Value> oldIterOperands = llvm::to_vector(op.getInitArgs());
  SmallVector<Value> newIterOperands = llvm::to_vector(op.getInitArgs());
  for (unsigned i = 0, oldI = 0, size = op.getInitArgs().size(); i < size;
       ++i, ++oldI) {
    if (!triton::isTensorPointerType(newIterOperands[i].getType()))
      continue;

    // Expand the tensor pointer into offsets
    assert(rewritedInfo.count(newIterOperands[i]));
    auto info = rewritedInfo[newIterOperands[i]];
    newIterOperands =
        generateNewOperands(newIterOperands, i, info.getOffsets());
    i += info.length() - 1;
    size += info.length() - 1;
  }

  // Rebuild the loop type
  auto newForOp = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                             op.getUpperBound(), op.getStep(),
                                             newIterOperands);

  // Create value mapping. Note that for tensor pointers, we use identity
  // mapping. It may refer to a value in the old loop, but we will rewrite it
  // later
  IRMapping mapping;
  for (unsigned i = 0, oldI = 0, sz = op.getInitArgs().size(); oldI < sz;
       ++i, ++oldI) {
    auto oldRegionIterArg = op.getRegionIterArg(oldI);
    if (triton::isTensorPointerType(oldRegionIterArg.getType())) {
      // Pass rewritten info inside
      assert(rewritedInfo.count(oldIterOperands[oldI]));
      auto info = rewritedInfo[oldIterOperands[oldI]];
      mapping.map(oldRegionIterArg, oldRegionIterArg);
      for (unsigned j = 0; j < info.length(); ++j)
        info.setOffset(j, newForOp.getRegionIterArg(i + j));
      rewritedInfo[oldRegionIterArg] = info;
      i += info.length() - 1;
    } else {
      mapping.map(oldRegionIterArg, newForOp.getRegionIterArg(i));
    }
  }
  mapping.map(op.getInductionVar(), newForOp.getInductionVar());

  // Clone body
  builder.setInsertionPointToStart(newForOp.getBody());
  for (auto &opInFor : *op.getBody()) {
    auto *newOp = builder.clone(opInFor, mapping);
    for (unsigned i = 0; i < opInFor.getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
  }

  // Replace later usages
  assert(op.getNumResults() == op.getInitArgs().size());
  for (unsigned i = 0, oldI = 0; oldI < op.getNumResults(); ++i, ++oldI) {
    auto oldResult = op.getResult(oldI);
    if (triton::isTensorPointerType(oldResult.getType())) {
      // Pack new offsets into rewritten info
      assert(rewritedInfo.count(oldIterOperands[oldI]));
      auto info = rewritedInfo[oldIterOperands[oldI]];
      for (unsigned j = 0; j < info.length(); ++j)
        info.setOffset(j, newForOp.getResult(i + j));
      i += info.length() - 1;
      rewritedInfo[oldResult] = info;
    } else {
      oldResult.replaceAllUsesWith(newForOp.getResult(i));
    }
  }

  // Erase later
  eraser.push(op);
  return newForOp;
}

Operation *
ConvertTensorPointerPass::rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                                         std::stack<Operation *> &eraser) {
  // Replace tensor pointers with offsets
  SmallVector<Value> newOperands = op->getOperands();
  for (unsigned i = 0, size = op.getNumOperands(); i < size; ++i) {
    if (!triton::isTensorPointerType(newOperands[i].getType()))
      continue;

    assert(rewritedInfo.count(newOperands[i]));
    auto info = rewritedInfo[newOperands[i]];
    newOperands = generateNewOperands(newOperands, i, info.getOffsets());
    i += info.length() - 1;
    size += info.length() - 1;
  }
  op->setOperands(newOperands);

  // No need to erase
  return nullptr;
}

Operation *
ConvertTensorPointerPass::rewriteOp(Operation *op,
                                    std::stack<Operation *> &eraser) {
  OpBuilder builder(op);

  // Rewrite `make_tensor_ptr` and `advance` and make a tensor of pointers
  // Rewriting functions return the next operation to visit, if there is no
  // next one, simply return `nullptr`
  if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
    return rewriteMakeTensorPtrOp(builder, makeTensorPtrOp, eraser);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
    return rewriteAdvanceOp(builder, advanceOp, eraser);
  } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
    return rewriteLoadStoreOp(builder, op, eraser);
  } else if (op->getDialect()->getNamespace() == "scf" ||
             op->getDialect()->getNamespace() == "cf") {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return rewriteIfOp(builder, ifOp, eraser);
    }
    if (!needRewrite(op))
      return op;

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return rewriteForOp(builder, forOp, eraser);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      return rewriteYieldOp(builder, yieldOp, eraser);
    } else {
      llvm_unreachable("Currently we only support tensor pointer usages "
                       "inside a `scf::ForOp` or `scf::IfOp`, others such as "
                       "`scf::WhileOp`, `cf::BranchOp` or `cf::CondBranchOp` "
                       "are not supported yet");
    }
  }

  // Otherwise return the original one
  return op;
}

void ConvertTensorPointerPass::visitOperation(Operation *op,
                                              std::stack<Operation *> &eraser) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      // We need an extra copy because erasing operations may break the
      // iterator behavior
      SmallVector<Operation *> blockCopy;
      for (auto &nestedOp : block)
        blockCopy.push_back(&nestedOp);

      // Rewrite and recursively visit
      for (auto &nestedOp : blockCopy) {
        if (auto newOp = rewriteOp(nestedOp, eraser))
          visitOperation(newOp, eraser);
      }
    }
  }
}

void ConvertTensorPointerPass::runOnOperation() {
  std::stack<Operation *> eraser;
  visitOperation(getOperation(), eraser);
  rewritedInfo.clear();
  while (!eraser.empty()) {
    auto op = eraser.top();
    eraser.pop();
    op->erase();
  }
}
