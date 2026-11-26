/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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

#include "TritonToStructured/PtrAnalysis.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "Utils/Utils.h"

#define DEBUG_TYPE "triton-to-structured-ptr-analysis"

namespace TritonToStructured {
using namespace mlir;
using namespace triton;

bool isMultiple(const OpFoldResult &dividend, const OpFoldResult &divisor) {
  auto staticDividend = getIntAttr(dividend);
  auto staticDivisor = getIntAttr(divisor);
  if (!staticDividend || !staticDivisor || staticDivisor.value() == 0) {
    return false;
  }
  return staticDividend.value() % staticDivisor.value() == 0;
}

bool isEqual(const OpFoldResult &ofr1, const OpFoldResult &ofr2) {
  auto staticOfr1 = getIntAttr(ofr1);
  auto staticOfr2 = getIntAttr(ofr2);
  return staticOfr1 == staticOfr2;
}

bool isLess(const OpFoldResult &ofr1, const OpFoldResult &ofr2) {
  auto staticOfr1 = getIntAttr(ofr1);
  auto staticOfr2 = getIntAttr(ofr2);
  // When sorting for permute, the value determined at runtime
  // is greater than the value determined at compile time.
  if (!staticOfr1 && !staticOfr2) {
    return false; // keep relative order (stable_sort)
  }
  if (!staticOfr1 && staticOfr2) {
    return false; // dynamic > static
  }
  if (staticOfr1 && !staticOfr2) {
    return true; // static < dynamic
  }
  return staticOfr1.value() < staticOfr2.value();
}

bool isGreater(const OpFoldResult &ofr1, const OpFoldResult &ofr2) {
  auto staticOfr1 = getIntAttr(ofr1);
  auto staticOfr2 = getIntAttr(ofr2);
  // When sorting for permute, the value determined at runtime
  // is greater than the value determined at compile time.
  if (!staticOfr1 && !staticOfr2) {
    return false; // keep relative order (stable_sort)
  }
  if (!staticOfr1 && staticOfr2) {
    return true; // dynamic > static
  }
  if (staticOfr1 && !staticOfr2) {
    return false; // static < dynamic
  }
  return staticOfr1.value() > staticOfr2.value();
}

void StateInfo::dump() const {
  llvm::dbgs() << "StateInfo: \n";
  llvm::dbgs() << "dimIndex = " << dimIndex << "\n";
  llvm::dbgs() << "shape = " << shape << "\n";
  llvm::dbgs() << "stride = " << stride << "\n";
}

void PtrState::dump() const {
  llvm::dbgs() << "PtrState: \n";
  llvm::dbgs() << "source:" << source << "\n";
  llvm::dbgs() << "scalar:" << offset << "\n";
  llvm::dbgs() << "size: [";
  for (auto size : sizes)
    llvm::dbgs() << size << ", ";
  llvm::dbgs() << "]\n";
  llvm::dbgs() << "shouldLinearize: " << shouldLinearize << "\n";
  llvm::dbgs() << "isPermuted: " << isPermuted << "\n";
  llvm::dbgs() << "isBlockPtr: " << isBlockPtr() << "\n";

  llvm::dbgs() << "permuteIds: [";
  for (auto id : permuteIds)
    llvm::dbgs() << id << ", ";
  llvm::dbgs() << "]\n";
  llvm::dbgs() << "order: [";
  for (auto id : order)
    llvm::dbgs() << id << ", ";
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "stateInfo:\n";
  llvm::dbgs() << "\n";
  for (auto info : stateInfo) {
    llvm::dbgs() << "-----------------------------------------\n";
    info.dump();
    llvm::dbgs() << "-----------------------------------------\n";
  }
}

bool PtrState::isEmpty() const {
  return (stateInfo.empty() && !source && !offset);
}

bool PtrState::isScalar() const {
  bool scalar = true;
  for (auto info : stateInfo) {
    auto staticStride = getIntAttr(info.stride);
    if (!staticStride.has_value() || staticStride.value() != 0)
      scalar = false;
  }
  return scalar && (offset || source);
}

bool PtrState::hasSource() const { return source != nullptr; }

bool PtrState::isBlockPtr() const { return !order.empty(); }

bool PtrState::isSameSizeAs(const PtrState &x) const {
  if (this->sizes.size() != x.sizes.size())
    return false;

  for (size_t i = 0; i < this->sizes.size(); ++i) {
    if (this->sizes[i] != x.sizes[i])
      return false;
  }
  return true;
}

void PtrState::updatePtrState(SmallVector<StateInfo> stateInfo,
                              SmallVector<OpFoldResult> sizes, Value source,
                              OpFoldResult offset, const Location loc,
                              OpBuilder &builder, bool shouldLinearize) {
  this->stateInfo = stateInfo;
  this->sizes = sizes;
  this->source = source;
  this->offset = offset;
  this->shouldLinearize = shouldLinearize;
  this->normalizeState(loc, builder);
}

void PtrState::normalizeState(const Location loc, OpBuilder &builder) {
  SmallVector<StateInfo> newStateInfo;
  auto zeroAttr = builder.getIndexAttr(0);

  // merge continuous zero strides
  // e.g., stride [0, 0, 1] shape [4, 32, 16]  --> stride [0, 1] shape [128, 16]
  for (auto it = this->stateInfo.begin(); it != this->stateInfo.end(); ++it) {
    while (it != this->stateInfo.end() && isZero(it->stride)) {
      auto newShape = it->shape;
      auto dimIndex = it->dimIndex;
      for (++it; it != this->stateInfo.end() && isZero(it->stride) &&
                 it->dimIndex == dimIndex;
           ++it) {
        newShape = mulOpFoldResult(newShape, it->shape, loc, builder,
                                   builder.getIndexType());
      }
      newStateInfo.emplace_back(zeroAttr, newShape, dimIndex);
    }
    if (it == this->stateInfo.end())
      break;
    // if the info is the only one with oriSize 1 in this dimension, skip it
    // e.g., stride [0, 1] shape [1, 128] sizes [1, 128] do not delete the first
    // info
    if (isOne(it->shape) && !isOne(sizes[it->dimIndex]))
      continue;
    newStateInfo.emplace_back(*it);
  }

  this->stateInfo = newStateInfo;
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      addptrOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting addptr");
    });
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Visit addptr operation: " << addptrOp << "\n";
  });

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Before visiting addptr operands: \n";
    llvm::dbgs() << "PtrState: \n";
    ptrState.dump();
    llvm::dbgs() << "OffsetState: \n";
    offsetState.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  if (!ptrState.source) {
    LLVM_DEBUG({
      addptrOp.emitError("ptr field should provide source / base pointer");
    });
    return failure();
  }
  return state.addState(ptrState, offsetState, addptrOp, builder);
}

LogicalResult
PtrAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  if (!state.isEmpty()) {
    makeTPtrOp.emitError(
        "PtrAnalysis: PtrState should be empty when visiting make_tensor_ptr");
    return failure();
  }
  if (makeTPtrOp.getOrder().empty()) {
    LLVM_DEBUG(makeTPtrOp->emitRemark(
        "PtrAnalysis: expect tt.make_tensor_ptr to have order field set"));
    return failure();
  }

  // Build:
  //   - stateInfo: per-dimension (stride, shape, dimIndex) of the parent tensor
  //   - sizes: original tensor shape of the block
  //   - dimOffsets: the offset to the block in the parent tensor
  state.source = makeTPtrOp.getBase();
  state.dimOffsets = makeTPtrOp.getOffsets();
  state.order = SmallVector<size_t>(makeTPtrOp.getOrder());

  auto resType = cast<triton::PointerType>(makeTPtrOp.getResult().getType());
  auto pointeeType = cast<ShapedType>(resType.getPointeeType());
  auto pointeeShape = pointeeType.getShape();
  const int64_t rank = pointeeType.getRank();

  SmallVector<StateInfo> newStateInfo;
  for (int64_t i = 0; i < rank; i++) {
    state.sizes.push_back(builder.getIndexAttr(pointeeShape[i]));
    newStateInfo.emplace_back(makeTPtrOp.getStrides()[i],
                              makeTPtrOp.getShape()[i], i);
  }
  state.stateInfo = newStateInfo;

  assert(state.isBlockPtr() &&
         "tt.make_tensor_ptr pointer state should describe a block pointer");

  return success();
}

bool PtrAnalysis::operandIsScalar(Value operand) {
  auto tensorType = dyn_cast<mlir::RankedTensorType>(operand.getType());
  auto elementType =
      tensorType ? tensorType.getElementType() : operand.getType();
  bool isScalar = true;
  if (tensorType) {
    for (size_t i = 0; i < tensorType.getRank() && isScalar; ++i) {
      isScalar = tensorType.getDimSize(i) == 1;
    }
  }
  return isScalar &&
         (isa<IntegerType>(elementType) || isa<IndexType>(elementType));
}

LogicalResult PtrAnalysis::initStateByScalar(Value operand, PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  OpFoldResult newOffset;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<StateInfo> newStateInfo;
  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand);
    newOffset = castOp.getResult();
  } else if (isa<IndexType>(operand.getType())) {
    newOffset = operand;
  } else {
    auto tensorType = dyn_cast<mlir::RankedTensorType>(operand.getType());
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto zeroAttr = builder.getIndexAttr(0);
    auto oneAttr = builder.getIndexAttr(1);
    SmallVector<mlir::Value> indices;
    for (size_t i = 0; i < tensorType.getRank(); ++i) {
      indices.push_back(index);
      newSizes.emplace_back(oneAttr);
      newStateInfo.emplace_back(zeroAttr, oneAttr, i);
    }
    auto extractedElement =
        builder.create<mlir::tensor::ExtractOp>(loc, operand, indices);
    newOffset = extractedElement.getResult();
  }
  state.updatePtrState(newStateInfo, newSizes, nullptr, newOffset, loc,
                       builder);
  return success();
}

LogicalResult PtrAnalysis::initStateByPointer(Value operand, PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  Value newSource;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<StateInfo> newStateInfo;

  if (auto op = operand.getDefiningOp()) {
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
      return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                builder);
    } else if (auto bitCastOp = dyn_cast<triton::BitcastOp>(op)) {
      newSource = operand;
    } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
      LLVM_DEBUG({
        op->emitWarning("Unexpected operand defining operation tts.make_tptr.");
      });
      return failure();
    } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
      newSource = operand;
    } else {
      LLVM_DEBUG({ op->emitWarning("PtrAnalysis: Unexpected operand."); });
      return failure();
    }
  } else {
    newSource = operand;
  }
  auto newOffset = builder.getIndexAttr(0);
  state.updatePtrState(newStateInfo, newSizes, newSource, newOffset, loc,
                       builder);
  return success();
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  auto loc = op->getLoc();

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "mulState: " << op << "\n";
  });

  if (!isEmpty()) {
    LLVM_DEBUG({
      op->emitError("PtrAnalysis: PtrState should be empty when multiplying");
    });
    return failure();
  }

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.hasSource() || rhsState.hasSource()) {
    LLVM_DEBUG({
      op->emitError("PtrAnalysis: do not support base inters in multiplying");
    });
    return failure();
  } else if (!lhsState.isScalar() && !rhsState.isScalar()) {
    // do not support both tensors are effectively non-scalar
    LLVM_DEBUG({
      op->emitError(
          "PtrAnalysis: only support multiplying pointer states when one of "
          "them represent a scalar");
    });
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->isScalar() && lhs->isScalar()) {
    std::swap(lhs, rhs);
  }

  SmallVector<StateInfo> newStateInfo;
  for (auto info : lhs->stateInfo) {
    OpFoldResult newStride = mulOpFoldResult(info.stride, rhs->offset, loc,
                                             builder, builder.getIndexType());
    newStateInfo.emplace_back(newStride, info.shape, info.dimIndex);
  }

  auto newOffset = mulOpFoldResult(lhsState.offset, rhsState.offset, loc,
                                   builder, builder.getIndexType());
  updatePtrState(newStateInfo, lhs->sizes, lhs->source, newOffset, loc, builder,
                 lhs->shouldLinearize);

  LLVM_DEBUG({
    llvm::dbgs() << "After mulState: \n";
    this->dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  return success();
}

LogicalResult PtrState::subState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  auto loc = op->getLoc();
  if (!isEmpty()) {
    LLVM_DEBUG({
      op->emitError("PtrAnalysis: PtrState should be empty when subtracting");
    });
    return failure();
  }

  if (lhsState.hasSource() && rhsState.hasSource()) {
    LLVM_DEBUG({
      op->emitError(
          "PtrAnalysis: do not support both sides have base pointers in sub");
    });
    return failure();
  }

  if (!rhsState.isScalar()) {
    LLVM_DEBUG({
      op->emitError("PtrAnalysis: only support sub when one of "
                    "them represents a scalar");
    });
    return failure();
  }

  auto newOffset =
      subOpFoldResult(lhsState.offset, rhsState.offset, loc, builder);
  updatePtrState(lhsState.stateInfo, lhsState.sizes, lhsState.source, newOffset,
                 loc, builder, lhsState.shouldLinearize);

  return success();
}

LogicalResult PtrState::addState(PtrState &lhsState, PtrState &rhsState,
                                 Operation *op, OpBuilder &builder) {
  auto loc = op->getLoc();

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "addState: " << op << "\n";
  });

  if (!isEmpty()) {
    LLVM_DEBUG({
      op->emitError("PtrAnalysis: PtrState should be empty when adding");
    });
    return failure();
  }
  if (!lhsState.isSameSizeAs(rhsState)) {
    LLVM_DEBUG({
      op->emitError(
          "PtrAnalysis: The original size of the addition should be the same");
    });
    return failure();
  }

  SmallVector<StateInfo> newStateInfo;
  auto lIt = lhsState.stateInfo.begin();
  auto rIt = rhsState.stateInfo.begin();
  while (lIt != lhsState.stateInfo.end() && rIt != rhsState.stateInfo.end()) {
    if (lIt->dimIndex != rIt->dimIndex) {
      auto newInfo = lIt->dimIndex < rIt->dimIndex ? *lIt++ : *rIt++;
      newStateInfo.emplace_back(newInfo);
      continue;
    }
    if (!isMultiple(lIt->shape, rIt->shape) &&
        !isMultiple(rIt->shape, lIt->shape)) {
      LLVM_DEBUG({
        llvm::dbgs() << "LHS PtrState: \n";
        lhsState.dump();
        llvm::dbgs() << "RHS PtrState: \n";
        rhsState.dump();
        llvm::dbgs() << "----------------------------------------------\n";
      });
      LLVM_DEBUG({
        op->emitError("PtrAnalysis: the add operation have incompatible sizes");
      });
      return failure();
    }

    auto newShape = minOpFoldResult(lIt->shape, rIt->shape, loc, builder);
    if ((isLess(newShape, lIt->shape) && !isZero(lIt->stride) ||
         isLess(newShape, rIt->shape) && !isZero(rIt->stride))) {
      LLVM_DEBUG({
        llvm::dbgs() << "LHS PtrState: \n";
        lhsState.dump();
        llvm::dbgs() << "RHS PtrState: \n";
        rhsState.dump();
        llvm::dbgs() << "----------------------------------------------\n";
      });
      LLVM_DEBUG({
        op->emitError("PtrAnalysis: the add operation have incompatible sizes."
                      "Valid dimensions are split.");
      });
      return failure();
    }

    auto newStride = addOpFoldResult(lIt->stride, rIt->stride, loc, builder,
                                     builder.getIndexType());
    newStateInfo.emplace_back(newStride, newShape, lIt->dimIndex);

    if (isEqual(lIt->shape, newShape))
      ++lIt;
    else
      lIt->shape = divOpFoldResult(lIt->shape, newShape, loc, builder);
    if (isEqual(rIt->shape, newShape))
      ++rIt;
    else
      rIt->shape = divOpFoldResult(rIt->shape, newShape, loc, builder);
  }

  while (rIt != rhsState.stateInfo.end()) {
    newStateInfo.push_back(*rIt++);
  }
  while (lIt != lhsState.stateInfo.end()) {
    newStateInfo.push_back(*lIt++);
  }

  auto newSource = source = lhsState.source ? lhsState.source : rhsState.source;
  auto newOffset = addOpFoldResult(lhsState.offset, rhsState.offset, loc,
                                   builder, builder.getIndexType());
  auto newShouldLinearize =
      lhsState.shouldLinearize || rhsState.shouldLinearize;
  auto newSizes = lhsState.sizes;

  updatePtrState(newStateInfo, newSizes, newSource, newOffset, loc, builder,
                 newShouldLinearize);

  LLVM_DEBUG({
    llvm::dbgs() << "After addState: \n";
    this->dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  return success();
}

triton::AddPtrOp PtrState::createAddPtrOp(OpBuilder &builder, Location loc) {
  SmallVector<int64_t> tensorSizes;
  SmallVector<OpFoldResult> tensorStrides;

  auto zeroAttr = builder.getIndexAttr(0);
  auto oneAttr = builder.getIndexAttr(1);

  for (auto id : permuteIds) {
    auto info = stateInfo[id];
    if (isZero(info.stride))
      continue;
    tensorStrides.emplace_back(info.stride);
    tensorSizes.emplace_back(getIntAttr(info.shape).value());
  }

  // load a scalar pointer
  if (tensorSizes.empty()) {
    Value offsetValue = materializeValue(builder, loc, offset);
    if (offsetValue.getType().isIndex()) {
      offsetValue = builder.create<arith::IndexCastOp>(
          loc, builder.getI32Type(), offsetValue);
    }
    auto addptrOp = builder.create<triton::AddPtrOp>(loc, source.getType(),
                                                     source, offsetValue);
    return addptrOp;
  }

  SmallVector<Value> cachedRange;
  auto ptrType = cast<triton::PointerType>(source.getType());
  auto ptrTensorType = RankedTensorType::get({tensorSizes}, ptrType);
  auto broadCastType =
      RankedTensorType::get({tensorSizes}, builder.getI32Type());

  if (tensorSizes.size() != tensorStrides.size()) {
    LLVM_DEBUG(
        {
          InFlightDiagnostic diag =
              emitError(loc)
              << "PtrAnalysis: inconsistent tensor sizes and strides";
        });
    return nullptr;
  }
  for (size_t i = 0; i < tensorSizes.size(); ++i) {
    // make range
    auto indexI32RowType =
        RankedTensorType::get({tensorSizes[i]}, builder.getI32Type());
    Value makeRangeOp = builder.create<triton::MakeRangeOp>(
        loc, indexI32RowType, 0, tensorSizes[i]);

    // multiply stride
    Value strideValue = materializeValue(builder, loc, tensorStrides[i]);
    if (strideValue.getType().isIndex()) {
      strideValue = builder.create<arith::IndexCastOp>(
          loc, builder.getI32Type(), strideValue);
    }
    Value splatStride =
        builder.create<triton::SplatOp>(loc, indexI32RowType, strideValue);
    auto rangeAfterMul =
        builder.create<arith::MulIOp>(loc, makeRangeOp, splatStride);

    // reshape
    Value expandedValue = rangeAfterMul;
    for (size_t j = 0; j < tensorSizes.size(); ++j) {
      if (j == i)
        continue;
      expandedValue =
          builder.create<triton::ExpandDimsOp>(loc, expandedValue, j);
    }

    // broadcast
    auto broadcastValue =
        builder.create<triton::BroadcastOp>(loc, broadCastType, expandedValue);
    cachedRange.push_back(broadcastValue);
  }

  // combine the cachedRange
  Value rangeAfterCombine = cachedRange[0];
  for (size_t i = 1; i < cachedRange.size(); ++i) {
    rangeAfterCombine =
        builder.create<arith::AddIOp>(loc, rangeAfterCombine, cachedRange[i]);
  }

  // addOffset
  Value addValue = materializeValue(builder, loc, offset);
  if (addValue.getType().isIndex()) {
    addValue =
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), addValue);
  }
  Value splatOffset =
      builder.create<triton::SplatOp>(loc, broadCastType, addValue);
  auto rangeAfterAdd =
      builder.create<arith::AddIOp>(loc, rangeAfterCombine, splatOffset);

  // addPtr
  Value splatPtr = builder.create<triton::SplatOp>(loc, ptrTensorType, source);
  auto addptrOp = builder.create<triton::AddPtrOp>(loc, ptrTensorType, splatPtr,
                                                   rangeAfterAdd);
  return addptrOp;
}

triton::MakeTensorPtrOp PtrState::createMakeTensorPtrOp(OpBuilder &builder,
                                                        Location loc) {
  SmallVector<Value> newShape;
  SmallVector<Value> newStrides;
  SmallVector<Value> newOffsets;
  SmallVector<int32_t> newBlkShape;
  SmallVector<int32_t> newOrder; // must be int32_t for MakeTensorPtrOp builder

  const size_t rank = order.size();
  if (rank == 0) {
    emitError(loc) << "PtrAnalysis: empty order in createMakeTensorPtrOp";
    return nullptr;
  }

  // iterate reversed safely: i = rank-1, ..., 0
  for (size_t i = rank; i-- > 0;) {
    size_t dim = order[i];

    if (dim >= stateInfo.size() || dim >= dimOffsets.size() ||
        dim >= sizes.size()) {
      emitError(loc)
          << "PtrAnalysis: invalid dim index in createMakeTensorPtrOp";
      return nullptr;
    }

    auto info = stateInfo[dim];

    newShape.push_back(materializeValue(builder, loc, info.shape));
    newStrides.push_back(materializeValue(builder, loc, info.stride));
    newOffsets.push_back(materializeValue(builder, loc, dimOffsets[dim]));

    auto blkSzOpt = getIntAttr(sizes[dim]);
    if (!blkSzOpt.has_value()) {
      emitError(loc) << "PtrAnalysis: dynamic block_shape is not supported for "
                        "tt.make_tensor_ptr";
      return nullptr;
    }
    newBlkShape.push_back(static_cast<int32_t>(blkSzOpt.value()));
    newOrder.push_back(static_cast<int32_t>(i));
  }

  return builder.create<triton::MakeTensorPtrOp>(
      loc, source, ValueRange(newShape), ValueRange(newStrides),
      ValueRange(newOffsets), newBlkShape, newOrder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Visit Mul operation: " << mulOp << "\n";
  });

  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  return state.mulState(lhsState, rhsState, mulOp, builder);
}

LogicalResult PtrAnalysis::visitOperandSub(arith::SubIOp subOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(subOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(subOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  return state.subState(lhsState, rhsState, subOp, builder);
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      rangeOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting make_range");
    });
    return failure();
  }

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  if (stride != 1) {
    LLVM_DEBUG({
      rangeOp.emitError(
          "PtrAnalysis: make_range op with stride != 1 is not supported");
    });
    return failure();
  }

  auto infoStride = builder.getIndexAttr(stride);
  auto size = builder.getIndexAttr(shape[0]);
  auto offset = builder.getIndexAttr(start);

  SmallVector<StateInfo> stateInfo;
  SmallVector<OpFoldResult> sizes;
  stateInfo.emplace_back(infoStride, size);
  sizes.emplace_back(size);

  state.updatePtrState(stateInfo, sizes, nullptr, offset, loc, builder);
  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      broadcastOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting broadcast");
    });
    return failure();
  }

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  if (!isa<ShapedType>(dst.getType())) {
    LLVM_DEBUG({
      broadcastOp.emitRemark(
          "PtrAnalysis: broadcast dst should be a shaped type");
    });
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  if (srcShape.size() != dstShape.size()) {
    LLVM_DEBUG({
      broadcastOp.emitRemark(
          "PtrAnalysis: broadcast src and dst should have the same rank");
    });
    return failure();
  }
  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (state.sizes.size() != dstShape.size()) {
    llvm::dbgs() << broadcastOp << "\n";
    state.dump();
    llvm::dbgs() << "dst.size = " << dstShape.size() << "\n";
    for (auto x : dstShape)
      llvm::dbgs() << x << ", ";
    llvm::dbgs() << "\n";
  }

  SmallVector<StateInfo> newStateInfo(state.stateInfo);
  SmallVector<OpFoldResult> newSizes;
  if (srcShape.size() != dstShape.size()) {
    LLVM_DEBUG({
      broadcastOp.emitRemark(
          "PtrAnalysis: unexpected state info size in broadcast");
    });
    return failure();
  }
  for (size_t i = 0; i < dstShape.size(); ++i) {
    newSizes.emplace_back(builder.getIndexAttr(dstShape[i]));
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i] && srcShape[i] == 1) {
      for (auto &info : newStateInfo) {
        if (info.dimIndex != i)
          continue;
        info.shape = builder.getIndexAttr(dstShape[i]);
      }
    } else {
      LLVM_DEBUG({
        broadcastOp.emitRemark("unexpected dimensions used in broadcast");
      });
      return failure();
    }
  }
  state.updatePtrState(newStateInfo, newSizes, state.source, state.offset, loc,
                       builder, state.shouldLinearize);
  return success();
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      splatOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting splat");
    });
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Visit SPLAT operation: " << splatOp << "\n";
  });

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "splat ptrState: \n";
    state.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  if (!state.isScalar()) {
    LLVM_DEBUG(
        { splatOp.emitRemark("PtrAnalysis: splat source should be scalar"); });
    return failure();
  }

  SmallVector<StateInfo> newStateInfo;
  SmallVector<OpFoldResult> newSizes;
  auto zeroAttr = builder.getIndexAttr(0);
  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (size_t i = 0; i < dstShape.size(); ++i) {
      auto currentSize = builder.getIndexAttr(dstShape[i]);
      newSizes.emplace_back(currentSize);
      newStateInfo.emplace_back(zeroAttr, currentSize, i);
    }
  } else {
    LLVM_DEBUG(
        { splatOp.emitRemark("PtrAnalysis: unsupported splat pattern"); });
    return failure();
  }
  state.updatePtrState(newStateInfo, newSizes, state.source, state.offset, loc,
                       builder, state.shouldLinearize);

  LLVM_DEBUG({
    llvm::dbgs() << "After SPLAT ptrState: \n";
    state.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });
  return success();
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      expandDimsOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting expand_dims");
    });
    return failure();
  }

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  SmallVector<StateInfo> newStateInfo(state.stateInfo);
  SmallVector<OpFoldResult> newSizes(state.sizes);
  size_t insertPos = 0;
  for (auto &info : newStateInfo) {
    if (info.dimIndex >= axis)
      ++info.dimIndex;
    if (info.dimIndex < axis)
      ++insertPos;
  }
  auto zeroAttr = builder.getIndexAttr(0);
  auto oneAttr = builder.getIndexAttr(1);
  StateInfo insertInfo(zeroAttr, oneAttr, axis);

  newStateInfo.insert(newStateInfo.begin() + insertPos, insertInfo);
  newSizes.insert(newSizes.begin() + axis, oneAttr);

  state.updatePtrState(newStateInfo, newSizes, state.source, state.offset, loc,
                       builder, state.shouldLinearize);
  return success();
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      op->emitError(
          "PtrAnalysis: PtrState should be empty when visiting const_splat");
    });
    return failure();
  }

  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  if (!attr.isSplat() || !isa<IntegerType>(elementType)) {
    LLVM_DEBUG(
        { op->emitError("PtrAnalysis: only support splat integer constant"); });
    return failure();
  }

  auto value = attr.getValues<IntegerAttr>()[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());

  auto resultShape = cast<ShapedType>(op.getResult().getType()).getShape();

  SmallVector<OpFoldResult> sizes;
  SmallVector<StateInfo> stateInfo;
  auto defaultAttr = builder.getIndexAttr(0);

  for (auto [i, shape] : llvm::enumerate(resultShape)) {
    auto shapeAttr = builder.getIndexAttr(shape);
    sizes.emplace_back(shapeAttr);
    stateInfo.emplace_back(defaultAttr, shapeAttr, i);
  }

  state.updatePtrState(stateInfo, sizes, nullptr, constAttr, loc, builder);
  return success();
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      extOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting extsi");
    });
    return failure();
  }

  if (visitOperand(extOp.getIn(), state, loc, builder).failed()) {
    return failure();
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      remOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting remsi");
    });
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "before VisitRemOperands \n";
    state.dump();
  });

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.isScalar() || rhsState.hasSource()) {
    LLVM_DEBUG({
      remOp.emitRemark("PtrAnalysis: only support cases when rhs of remainder "
                       "contains scalar");
    });
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  bool hasAnnotation = optimizeDynamicOffset;

  auto zeroAttr = builder.getIndexAttr(0);
  auto oneAttr = builder.getIndexAttr(1);
  auto divisorAttr = rhsState.offset;

  auto staticOffset = getIntAttr(state.offset);
  if ((!staticOffset.has_value() || !isMultiple(state.offset, divisorAttr)) &&
      !hasAnnotation) {
    LLVM_DEBUG({
      remOp.emitRemark(
          "PtrAnalysis: dynamic offset before REMSI, adding annotation");
    });
    return failure();
  }

  if (!getIntAttr(divisorAttr).has_value()) {
    LLVM_DEBUG({
      remOp.emitError("PtrAnalysis: do not support dynamix divisor in REMSI.");
    });
    return failure();
  }

  SmallVector<StateInfo> newStateInfo;
  for (auto info : state.stateInfo) {
    if (!getIntAttr(info.stride).has_value()) {
      LLVM_DEBUG({
        remOp.emitError(
            "PtrAnalysis: do not support dynamix stride before REMSI.");
      });
      return failure();
    }

    if (!isMultiple(divisorAttr, info.shape) &&
        !isMultiple(info.shape, divisorAttr)) {
      LLVM_DEBUG({
        remOp.emitError(
            "PtrAnalysis: do not support dynamix stride before REMSI.");
      });
    }

    if (isMultiple(info.stride, divisorAttr)) {
      newStateInfo.emplace_back(zeroAttr, info.shape, info.dimIndex);
    } else if (isMultiple(divisorAttr, info.stride)) {
      auto contiguousSize =
          divOpFoldResult(divisorAttr, info.stride, loc, builder);
      contiguousSize =
          minOpFoldResult(contiguousSize, info.shape, loc, builder);
      auto nonContiguousSize =
          divOpFoldResult(info.shape, contiguousSize, loc, builder);

      auto staticNonContiguousSize = getIntAttr(nonContiguousSize);
      if (!staticNonContiguousSize.has_value()) {
        LLVM_DEBUG({
          remOp.emitError(
              "PtrAnalysis: do not support dynamix size before REMSI.");
        });
        return failure();
      }

      if (staticNonContiguousSize.value() > 1)
        newStateInfo.emplace_back(zeroAttr, nonContiguousSize, info.dimIndex);

      newStateInfo.emplace_back(info.stride, contiguousSize, info.dimIndex);
    } else {
      LLVM_DEBUG({
        remOp.emitError("PtrAnalysis: stride that are not divisible by REMSI "
                        "are not allowed "
                        "to precede REMSI");
      });
      return failure();
    }
  }

  auto newOffset = remOpFoldResult(state.offset, divisorAttr, loc, builder);
  state.updatePtrState(newStateInfo, state.sizes, state.source, newOffset, loc,
                       builder, true);

  LLVM_DEBUG({
    llvm::dbgs() << "after VisitRemOperands \n";
    state.dump();
  });

  return success();
}

LogicalResult PtrAnalysis::visitOperandDiv(arith::DivSIOp divOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  if (!state.isEmpty()) {
    LLVM_DEBUG({
      divOp.emitError(
          "PtrAnalysis: PtrState should be empty when visiting divsi");
    });
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Visit DIVSI operation: " << divOp << "\n";
  });

  PtrState rhsState;
  if (visitOperand(divOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.isScalar() || rhsState.hasSource()) {
    LLVM_DEBUG({
      divOp.emitRemark("PtrAnalysis: only support cases when rhs of remainder "
                       "contains scalar");
    });
    return failure();
  }

  if (visitOperand(divOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  bool hasAnnotation = optimizeDynamicOffset;

  auto staticMultipleOf = extractDivisibilityFromOpFoldResult(state.offset);
  if (!hasAnnotation && staticMultipleOf.has_value()) {
    auto attr = builder.getIndexAttr(staticMultipleOf.value());
    hasAnnotation = isMultiple(attr, rhsState.offset);
  }

  // add divState
  auto zeroAttr = builder.getIndexAttr(0);
  auto oneAttr = builder.getIndexAttr(1);
  auto divisorAttr = rhsState.offset;

  auto staticOffset = getIntAttr(state.offset);
  if ((!staticOffset.has_value() || !isMultiple(state.offset, divisorAttr)) &&
      !hasAnnotation) {
    LLVM_DEBUG({
      divOp.emitRemark(
          "PtrAnalysis: dynamic offset before DIVSI, adding annotation");
    });
    return failure();
  }

  if (!getIntAttr(divisorAttr).has_value()) {
    LLVM_DEBUG({
      divOp.emitError("PtrAnalysis: do not support dynamix divisor in DIVSI.");
    });
    return failure();
  }

  SmallVector<StateInfo> newStateInfo;
  for (auto info : state.stateInfo) {
    auto staticStride = getIntAttr(info.stride);
    if (!staticStride.has_value()) {
      LLVM_DEBUG({
        divOp.emitError(
            "PtrAnalysis: do not support dynamix stride before DIVSI.");
      });
      return failure();
    }

    if (!isMultiple(divisorAttr, info.shape) &&
        !isMultiple(info.shape, divisorAttr)) {
      LLVM_DEBUG({
        divOp.emitError(
            "PtrAnalysis: do not support dynamix stride before DivSI.");
      });
    }

    if (isMultiple(info.stride, divisorAttr)) {
      auto newStride = divOpFoldResult(info.stride, divisorAttr, loc, builder);
      newStateInfo.emplace_back(newStride, info.shape, info.dimIndex);
    } else if (isMultiple(divisorAttr, info.stride)) {
      auto nonContiguousSize =
          divOpFoldResult(divisorAttr, info.stride, loc, builder);
      nonContiguousSize =
          minOpFoldResult(nonContiguousSize, info.shape, loc, builder);
      auto contiguousSize =
          divOpFoldResult(info.shape, nonContiguousSize, loc, builder);

      auto staticContiguousSize = getIntAttr(contiguousSize);
      if (!staticContiguousSize.has_value()) {
        LLVM_DEBUG({
          divOp.emitError(
              "PtrAnalysis: do not support dynamix size before DIVSI.");
        });
        return failure();
      }

      if (staticContiguousSize.value() != 0)
        newStateInfo.emplace_back(oneAttr, contiguousSize, info.dimIndex);

      newStateInfo.emplace_back(zeroAttr, nonContiguousSize, info.dimIndex);
    } else {
      LLVM_DEBUG({
        divOp.emitError("PtrAnalysis: stride that are not divisible by DIVSI "
                        "are not allowed "
                        "to precede DIVSI");
      });
      return failure();
    }
  }

  auto newOffset = divOpFoldResult(state.offset, divisorAttr, loc, builder);
  state.updatePtrState(newStateInfo, state.sizes, state.source, newOffset, loc,
                       builder, true);

  LLVM_DEBUG({
    llvm::dbgs() << "after VisitDivOperands \n";
    state.dump();
  });
  return success();
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed())
    return failure();
  return state.addState(lhsState, rhsState, addOp, builder);
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {
  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (operandIsScalar(operand)) {
    return initStateByScalar(operand, state, loc, builder);
  }

  if (isa<triton::PointerType>(operand.getType())) {
    return initStateByPointer(operand, state, loc, builder);
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return visitOperandSub(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return visitOperandDiv(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::LoadOp>()) {
    LLVM_DEBUG({
      op.emitRemark("TritonToStructured: Invalid dynamic offset"
                    "The load operation's offset cannot be derived from "
                    "another load result.");
    });
    return failure();
  } else if (auto op = operand.getDefiningOp<arith::FPToSIOp>()) {
    LLVM_DEBUG({
      op.emitWarning("IllegalTypeConversionInAddressCalculation"
                     "float-to-int precision conversion is not supported "
                     "during address computation.");
      llvm::dbgs() << "Operand: \n";
      operand.dump();
      llvm::dbgs() << "----------------------------------------------\n";
    });
    return failure();
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      LLVM_DEBUG({
        llvm::dbgs() << "TritonToStructured: Pointer analysis is not supported "
                        "for input parameters\n";
      });
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    auto op = operand.getDefiningOp();
    LLVM_DEBUG({
      op->emitWarning("TritonToStructured: encountered addptr operand produced "
                      "by an unsupported operation");
      llvm::dbgs() << "Operand: \n";
      operand.dump();
      llvm::dbgs() << "----------------------------------------------\n";
    });
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  auto maketptrOp = state.createAddPtrOp(builder, op.getLoc());
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp.getResult());

  return success();
}

void PtrState::analyzePermute() {
  const size_t n = stateInfo.size();
  generateOriginPermuteIds();

  if (n <= 1)
    return;

  // ============================================================
  // === 1. make_tensor_ptr (block ptr): order-based ===
  // Rule: permute if order is NOT canonical (i.e. strictly decreasing,
  // representing inner-to-outer memory layout priority, e.g. [n-1, ..., 0])
  //
  // NOTE: order describes memory layout priority, not axis permutation.
  // Do NOT translate order into permuteIds.
  // ============================================================
  if (isBlockPtr()) {
    for (size_t i = 0; i + 1 < n; ++i) {
      if (order[i] <= order[i + 1]) {
        isPermuted = true;
        break;
      }
    }
    return;
  }

  // ============================================================
  // === 2. addptr: stride-based permuteIds + contiguous-axis ===
  // Rule: permute if contiguous axes increased
  //
  // analyze constraints: must have at least one static stride
  // ============================================================
  bool hasStatic = false;
  for (auto &s : stateInfo) {
    if (getIntAttr(s.stride).has_value()) {
      hasStatic = true;
      break;
    }
  }
  if (!hasStatic) {
    return;
  }
  auto isIdentity = [&](ArrayRef<size_t> perm) -> bool {
    for (size_t i = 0; i < perm.size(); ++i)
      if (perm[i] != i)
        return false;
    return true;
  };

  std::stable_sort(permuteIds.begin(), permuteIds.end(),
                   [&](size_t a, size_t b) {
                     return isGreater(stateInfo[a].stride, stateInfo[b].stride);
                   });

  // If already in canonical axis order, do not permute.
  if (isIdentity(permuteIds)) {
    return;
  }

  // Tail axis must be physically contiguous (stride == 1),
  // otherwise addptr-based permutation is invalid.
  auto tailStride = getIntAttr(stateInfo[permuteIds.back()].stride);
  if (!tailStride.has_value() || tailStride.value() != 1) {
    generateOriginPermuteIds();
    return;
  }

  // Compute new contiguous axes count using the permuted stateInfo.
  SmallVector<StateInfo> newStateInfo;
  newStateInfo.reserve(n);
  for (size_t id : permuteIds) {
    newStateInfo.push_back(stateInfo[id]);
  }
  const size_t oldContig = countContiguousAxes(stateInfo);
  const size_t newContig = countContiguousAxes(newStateInfo);
  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "after analyzePermute:\n"
                 << "oldContig: " << oldContig << "\n"
                 << "newContig: " << newContig << "\n";
    dump();
  });
  // only permute if contiguous axes increased
  if (newContig > oldContig) {
    isPermuted = true;
    return;
  }

  // otherwise: no permute
  generateOriginPermuteIds();
}

std::optional<int32_t>
extractDivisibilityFromOpFoldResult(mlir::OpFoldResult ofr) {
  auto value = dyn_cast<mlir::Value>(ofr);
  if (!value) {
    return std::nullopt;
  }
  auto defOp = value.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }

  auto divisibilityAttr = defOp->getAttr("tt.divisibility");
  if (!divisibilityAttr) {
    return std::nullopt;
  }

  auto denseAttr = dyn_cast<mlir::DenseIntElementsAttr>(divisibilityAttr);
  if (!denseAttr || denseAttr.empty()) {
    return std::nullopt;
  }

  return denseAttr.getValues<int32_t>()[0];
}

void PtrState::generateOriginPermuteIds() {
  permuteIds.clear();
  isPermuted = false;
  for (size_t i = 0; i < stateInfo.size(); ++i) {
    permuteIds.emplace_back(i);
  }
  return;
}

// Formula of "contiguous axes"
// - axis i is contiguous if stride[i] == product(shape[0..i-1]).
// We only prove contiguity from the rightmost axis outward.
// Rules implemented
// - Start from the rightmost axis. It is contiguous only if stride == 1
// (static).
// - Then move left; expectedStride multiplies by the static shape of the axis
// to the right.
// - If we encounter any dynamic stride or dynamic shape needed for
// expectedStride,
//   stop (cannot prove further) and return the count accumulated so far.
// Examples
//  - shape=[2,3,4,5], stride=[A,B,2,C] => rightmost stride is dynamic =>
//  count=0
//  - shape=[2,3,4,5], stride=[A,B,C,1] => rightmost stride=1 => count=1
//  - shape=[2,3,4,5], stride=[60,20,5,1] => count=4
size_t PtrState::countContiguousAxes(SmallVector<StateInfo> stateInfo) const {
  if (stateInfo.empty())
    return 0;
  int64_t expected = 1;
  size_t cnt = 0;
  // iterate reversed safely: i = stateInfo.size()-1, ..., 0
  for (size_t i = stateInfo.size(); i-- > 0;) {
    auto stride = getIntAttr(stateInfo[i].stride);
    if (!stride.has_value())
      break;
    if (stride.value() != expected)
      break;
    ++cnt;
    // Update expected for the next (outer) axis: expected *= shape[i]
    auto shape = getIntAttr(stateInfo[i].shape);
    if (!shape.has_value())
      break;
    expected *= shape.value();
  }
  return cnt;
}
} // namespace TritonToStructured
