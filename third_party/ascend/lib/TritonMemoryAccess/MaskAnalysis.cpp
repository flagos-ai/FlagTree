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

#include "TritonMemoryAccess/LoadStoreMaskAnalysis.h"
#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonMemoryAccess/OpFoldResultUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "mask-analysis"

namespace mlir {

namespace triton {

namespace {

template <typename MemAccOpTy>
std::optional<MaskState> runMaskAnalysisImpl(MemAccOpTy op,
                                             OpBuilder &builder) {
  auto mask = op.getMask();
  if (!mask) {
    return std::nullopt;
  }

  PatternRewriter::InsertionGuard insertGuard(builder);
  builder.setInsertionPoint(op);

  MaskState mstate;
  if (mstate.parse(mask, op.getLoc(), builder).failed()) {
    return std::nullopt;
  }
  return mstate;
}

bool isZeroMaskConstant(const OpFoldResult &value) {
  // Keep the actual mlir::isZeroInteger implementation used by the source
  // baseline: a zero IntegerAttr or a Value matched as a constant integer.
  if (auto constant = getConstantIntValue(value))
    return *constant == 0;
  return false;
}

static Value unwrapRuntimeExtentUnsignedOperand(Value operand,
                                                const Location &loc,
                                                OpBuilder &builder) {
  if (auto extend = operand.getDefiningOp<arith::ExtUIOp>())
    return extend.getIn();

  auto splat = operand.getDefiningOp<triton::SplatOp>();
  if (!splat)
    return Value();
  auto extend = splat.getSrc().getDefiningOp<arith::ExtUIOp>();
  if (!extend)
    return Value();

  auto resultType = dyn_cast<RankedTensorType>(splat.getType());
  if (!resultType || !isa<IntegerType>(extend.getIn().getType()))
    return Value();
  auto narrowedType =
      RankedTensorType::get(resultType.getShape(), extend.getIn().getType(),
                            resultType.getEncoding());
  return builder.create<triton::SplatOp>(loc, narrowedType, extend.getIn());
}

// Follow an i1 tensor to its init only when all corresponding loop edges
// forward the same value. A while may permute condition/result slots, so do
// not assume its before and after argument numbers are interchangeable.
Value getInvariantLoopMaskInit(BlockArgument argument) {
  Operation *parent = argument.getOwner()->getParentOp();
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    if (argument == loop.getInductionVar())
      return {};
    unsigned slot = argument.getArgNumber() - 1;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return yield.getOperand(slot) == argument ? loop.getInitArgs()[slot]
                                              : Value();
  }
  if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
    auto condition = loop.getConditionOp();
    auto yield = loop.getYieldOp();
    BlockArgument before, after;
    if (argument.getOwner() == &loop.getBefore().front()) {
      before = argument;
      after = dyn_cast<BlockArgument>(yield.getOperand(before.getArgNumber()));
      if (!after || after.getOwner() != &loop.getAfter().front() ||
          condition.getArgs()[after.getArgNumber()] != before)
        return {};
    } else {
      after = argument;
      before =
          dyn_cast<BlockArgument>(condition.getArgs()[after.getArgNumber()]);
      if (!before || before.getOwner() != &loop.getBefore().front() ||
          yield.getOperand(before.getArgNumber()) != after)
        return {};
    }
    return loop.getInits()[before.getArgNumber()];
  }
  return {};
}

} // namespace

OpFoldResult MaskState::clampToNonNegativeIndex(const OpFoldResult value,
                                                const Location &loc,
                                                OpBuilder &builder,
                                                bool clampDynamic) const {
  if (auto cst = getConstantIntValue(value)) {
    return builder.getIndexAttr(std::max<int64_t>(0, *cst));
  }

  // Only minStates needs the dynamic clamp (disjoint intersection -> negative
  // extent); the cmp paths are already non-negative, and clamping them
  // regressed the atomic max/min UT.
  if (clampDynamic) {
    return maxOpFoldResult(value, builder.getIndexAttr(0), loc, builder);
  }
  return value;
}

LogicalResult MaskState::parse(Value operand, const Location &loc,
                               OpBuilder &builder) {
  Type operandType = operand.getType();
  if (isa<IntegerType>(operandType) || operandType.isIndex()) {
    return parseIntScalar(operand, loc, builder);
  }

  if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
    auto parentOp = blockArgument.getOwner()->getParentOp();
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (type && type.getElementType().isInteger(1)) {
        Value init = getInvariantLoopMaskInit(blockArgument);
        return init ? parse(init, loc, builder) : failure();
      }
      OpOperand *initArgOperand = loopOp.getTiedLoopInit(blockArgument);
      if (initArgOperand) {
        if (!isa<ShapedType>(operand.getType()))
          return failure();

        // Parse the init value to get the base range structure
        if (failed(parse(initArgOperand->get(), loc, builder)))
          return failure();

        // Only scf.for loops are handled
        auto forOp = dyn_cast<scf::ForOp>(parentOp);
        if (!forOp)
          return failure();

        unsigned slot = blockArgument.getArgNumber() - 1;
        if (slot >= forOp.getYieldedValues().size())
          return failure();
        Value yielded = forOp.getYieldedValues()[slot];
        if (yielded == operand) {
          // iter_arg is unchanged across iterations: the init value is the
          // current value, so the parsed state needs no adjustment.
          return success();
        }

        // Detect yield == iter_arg + const_tensor (or const_tensor + iter_arg).
        auto addOp = yielded.getDefiningOp<arith::AddIOp>();
        if (!addOp)
          return failure();
        Value incrementValue;
        if (addOp.getLhs() == operand)
          incrementValue = addOp.getRhs();
        else if (addOp.getRhs() == operand)
          incrementValue = addOp.getLhs();
        else
          return failure();

        // The increment must be a splat integer constant tensor or a scalar
        // integer constant (i.e. iteration-independent).
        auto constOp = incrementValue.getDefiningOp<arith::ConstantOp>();
        if (!constOp)
          return failure();
        int64_t increment = 0;
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
          if (!denseAttr.isSplat() ||
              !isa<IntegerType>(denseAttr.getElementType()))
            return failure();
          increment = denseAttr.getSplatValue<IntegerAttr>().getInt();
        } else if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          increment = intAttr.getInt();
        } else {
          return failure();
        }

        // The iteration count n = (iv - lb) / step relates the induction
        // variable to the per-iteration increment: current = init + n*delta.
        // lb/step may be dynamic; cast to index and compute at runtime.
        FailureOr<Value> ivIndex = castIntegerLike(
            builder, loc, forOp.getInductionVar(), builder.getIndexType());
        if (failed(ivIndex))
          return failure();

        Value iterCount = *ivIndex;

        // iterCount = iv - lb  (handle dynamic or constant lb)
        auto lb = getConstantIntValue(forOp.getLowerBound());
        if (!lb || *lb != 0) {
          FailureOr<Value> lbIndex = castIntegerLike(
              builder, loc, forOp.getLowerBound(), builder.getIndexType());
          if (failed(lbIndex))
            return failure();
          iterCount = builder.create<arith::SubIOp>(loc, iterCount, *lbIndex);
        }

        // iterCount = (iv - lb) / step  (handle dynamic or constant step)
        auto stepCst = getConstantIntValue(forOp.getStep());
        if (stepCst && *stepCst == 0)
          return failure();
        if (!stepCst || *stepCst != 1) {
          FailureOr<Value> stepIndex = castIntegerLike(
              builder, loc, forOp.getStep(), builder.getIndexType());
          if (failed(stepIndex))
            return failure();
          iterCount =
              builder.create<arith::DivSIOp>(loc, iterCount, *stepIndex);
        }

        OpFoldResult offset =
            mulOpFoldResult(iterCount, builder.getIndexAttr(increment), loc,
                            builder, builder.getIndexType());
        if (!offset)
          return failure();

        if (this->start && this->end) {
          this->start = addOpFoldResult(this->start, offset, loc, builder,
                                        builder.getIndexType());
          this->end = addOpFoldResult(this->end, offset, loc, builder,
                                      builder.getIndexType());
          if (!this->start || !this->end)
            return failure();
        } else if (this->scalar) {
          this->scalar = addOpFoldResult(this->scalar, offset, loc, builder,
                                         builder.getIndexType());
          if (!this->scalar)
            return failure();
        } else {
          return failure();
        }
        return success();
      }
    }
  }

  auto definingOp = operand.getDefiningOp();
  if (!definingOp)
    return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "[MaskState]==> parse op\n"
                 << *definingOp << "\n[MaskState]<==\n";
  });
  return TypeSwitch<Operation *, LogicalResult>(definingOp)
      .Case<arith::ConstantOp>(
          [&](auto op) { return this->parseConstant(op, loc, builder); })
      .Case<arith::AddIOp>(
          [&](auto op) { return this->parseAdd(op, loc, builder); })
      .Case<arith::SubIOp>(
          [&](auto op) { return this->parseSub(op, loc, builder); })
      .Case<arith::AndIOp>(
          [&](auto op) { return this->parseAnd(op, loc, builder); })
      .Case<arith::CmpIOp>(
          [&](auto op) { return this->parseCmp(op, loc, builder); })
      .Case<triton::MakeRangeOp>(
          [&](auto op) { return this->parseMakeRange(op, loc, builder); })
      .Case<triton::BroadcastOp>(
          [&](auto op) { return this->parseBroadcast(op, loc, builder); })
      .Case<triton::SplatOp>(
          [&](auto op) { return this->parseSplat(op, loc, builder); })
      .Case<triton::ExpandDimsOp>(
          [&](auto op) { return this->parseExpandDims(op, loc, builder); })
      .Case<arith::ExtSIOp>(
          [&](auto op) { return this->parse(op.getIn(), loc, builder); })
      .Case<arith::DivSIOp>(
          [&](auto op) { return this->parseDiv(op, loc, builder); })
      .Case<arith::SelectOp>(
          [&](auto op) { return this->parseSel(op, loc, builder); })
      .Case<tensor::InsertOp>(
          [&](auto op) { return this->parseInsert(op, loc, builder); })
      .Default([&](Operation *op) { return failure(); });
}

// extractSlice
tensor::ExtractSliceOp MaskState::getExtractSlice(Value source,
                                                  const Location &loc,
                                                  OpBuilder &builder) const {
  auto sourceRType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  auto dstRType = tensor::ExtractSliceOp::inferResultType(sourceRType, offsets,
                                                          dims, strides);
  return builder.create<tensor::ExtractSliceOp>(loc, dstRType, source, offsets,
                                                dims, strides);
}

tensor::ExtractSliceOp MaskState::getExtractSlice(
    Value source, const Location &loc, OpBuilder &builder,
    SmallVector<OpFoldResult> offsets, SmallVector<OpFoldResult> dims) const {
  auto sourceRType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  auto dstRType = tensor::ExtractSliceOp::inferResultType(sourceRType, offsets,
                                                          dims, strides);
  return builder.create<tensor::ExtractSliceOp>(loc, dstRType, source, offsets,
                                                dims, strides);
}

// insertSlice
tensor::InsertSliceOp MaskState::getInsertSlice(Value source, Value dest,
                                                const Location &loc,
                                                OpBuilder &builder) const {
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  return builder.create<tensor::InsertSliceOp>(loc, source, dest, offsets, dims,
                                               strides);
}

tensor::InsertSliceOp
MaskState::getInsertSlice(Value source, Value dest, const Location &loc,
                          OpBuilder &builder, SmallVector<OpFoldResult> offsets,
                          SmallVector<OpFoldResult> dims) const {
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  return builder.create<tensor::InsertSliceOp>(loc, source, dest, offsets, dims,
                                               strides);
}

memref::SubViewOp MaskState::getSubview(Value source, const Location &loc,
                                        OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  int64_t rank = sourceType.getRank();
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

  SmallVector<OpFoldResult> fixedOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> fixedDims(dims.begin(), dims.end());
  fixedOffsets.resize(rank, builder.getIndexAttr(0));
  fixedDims.resize(rank, builder.getIndexAttr(1));

  auto dstType = memref::SubViewOp::inferResultType(sourceType, fixedOffsets,
                                                    fixedDims, strides);
  return builder.create<memref::SubViewOp>(
      loc, cast<MemRefType>(dstType), source, fixedOffsets, fixedDims, strides);
}

// In llvm later version, for each dimension, assert that:
// 0 <= offset < dim_size
// 0 <= offset + (size - 1) *stride < dim_size
// To adatpt to this change, add this verification to avoid llvm assert error
// And currently, in the function calling scenario, the stride coefficient is
// always 1
bool MaskState::isMemrefSubviewValid(Value source, OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  int64_t rank = sourceType.getRank();

  SmallVector<OpFoldResult> fixedOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> fixedDims(dims.begin(), dims.end());
  fixedOffsets.resize(rank, builder.getIndexAttr(0));
  fixedDims.resize(rank, builder.getIndexAttr(1));

  for (int64_t i = 0; i < rank; ++i) {
    int64_t sourceSize = sourceType.getDimSize(i);
    if (ShapedType::isDynamic(sourceSize))
      continue;
    std::optional<int64_t> offsetVal =
        mlir::getConstantIntValue(fixedOffsets[i]);
    std::optional<int64_t> dimVal = mlir::getConstantIntValue(fixedDims[i]);
    if (offsetVal.has_value() && dimVal.has_value()) {
      if (offsetVal.value() >= sourceSize || offsetVal.value() < 0) {
        LLVM_DEBUG({
          llvm::dbgs() << "MemrefSubview offset check faied at dim " << i
                       << "sourceSize :" << sourceSize
                       << "offsetVal :" << offsetVal << "\n";
        });
        return false;
      }

      int64_t computedEnd = offsetVal.value() + dimVal.value();
      if (computedEnd > sourceSize) {
        LLVM_DEBUG({
          llvm::dbgs() << "MemrefSubview offset end check faied at dim " << i
                       << "dimVal :" << dimVal << "offsetVal :" << offsetVal
                       << "\n";
        });
        return false;
      }
    }
  }
  return true;
}

static memref::SubViewOp createSubview(Value src, const Location &loc,
                                       OpBuilder &builder,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return builder.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), src,
                                           offsets, sizes, strides);
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar,
                                        const Location &loc,
                                        OpBuilder &builder) {
  start = addOpFoldResult(state.start, scalar, loc, builder,
                          builder.getIndexType());
  end =
      addOpFoldResult(state.end, scalar, loc, builder, builder.getIndexType());
  dims = state.dims;
  offsets = state.offsets;

  bool allDimsOne = llvm::all_of(state.dims, [](OpFoldResult dim) {
    return getConstantIntValue(dim).value() == std::optional<int64_t>(1);
  });
  if (allDimsOne) {
    this->scalar = this->start;
  }

  return success();
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }
  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar) {
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  } else {
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
  }
}

LogicalResult MaskState::subStateScalar(const MaskState &state,
                                        const OpFoldResult scalar,
                                        const Location &loc,
                                        OpBuilder &builder) {
  start = subOpFoldResult(state.start, scalar, loc, builder);
  end = subOpFoldResult(state.end, scalar, loc, builder);
  if (!start || !end)
    return failure();
  dims = state.dims;
  offsets = state.offsets;

  bool allDimsOne = llvm::all_of(state.dims, [](OpFoldResult dim) {
    return getConstantIntValue(dim).value() == std::optional<int64_t>(1);
  });
  if (allDimsOne) {
    this->scalar = this->start;
  }

  return success();
}

LogicalResult MaskState::subStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  // Contiguous mask indices only keep a monotonic range when subtracting a
  // scalar from a range: (arange - C). scalar - range is rejected.
  if (!lhsState.scalar && rhsState.scalar)
    return subStateScalar(lhsState, rhsState.scalar, loc, builder);

  InFlightDiagnostic diag =
      emitWarning(loc)
      << "Unsupported subi for continuous mask: only (range - scalar) is "
         "supported";
  return failure();
}

LogicalResult MaskState::divStateScalar(const MaskState &state,
                                        const OpFoldResult scalar,
                                        const Location &loc,
                                        OpBuilder &builder) {
  start = divOpFoldResult(state.start, scalar, loc, builder);
  end = divOpFoldResult(state.end, scalar, loc, builder);
  dims = state.dims;
  offsets = state.offsets;
  return success();
}

LogicalResult MaskState::divStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  if (!lhsState.scalar && rhsState.scalar) {
    if (isZeroMaskConstant(rhsState.scalar)) {
      InFlightDiagnostic diag =
          emitError(loc)
          << "Unsupported scenario where rhs is zero constant in divide!";
      return failure();
    }

    return divStateScalar(lhsState, rhsState.scalar, loc, builder);
  }

  InFlightDiagnostic diag = emitWarning(loc)
                            << "Supported scenario where only rhs is a scalar";
  return failure();
}

LogicalResult MaskState::minStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  if (lhsState.getRank() != rhsState.getRank()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unexpected case where lhs and rhs have different ranks";
    return failure();
  }

  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsOffset = lhsState.offsets[i];
    auto rhsOffset = rhsState.offsets[i];
    auto newOffset = maxOpFoldResult(lhsOffset, rhsOffset, loc, builder);
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    auto lhsEnd = addOpFoldResult(lhsOffset, lhsDim, loc, builder,
                                  builder.getIndexType());
    auto rhsEnd = addOpFoldResult(rhsOffset, rhsDim, loc, builder,
                                  builder.getIndexType());
    auto newEnd = minOpFoldResult(lhsEnd, rhsEnd, loc, builder);
    auto newDim = subOpFoldResult(newEnd, newOffset, loc, builder);
    auto clampedNewDim =
        clampToNonNegativeIndex(newDim, loc, builder, /*clampDynamic=*/true);

    offsets.push_back(newOffset);
    dims.push_back(clampedNewDim);
  }
  return success();
}

// Helper func for MaskState::parse()
LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location &loc,
                                       OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "All elements must share a single integer constant value");

    if (elementType.isInteger(1) &&
        isa<ShapedType>(constOp.getValue().getType())) {
      auto shapedType = cast<ShapedType>(constOp.getValue().getType());
      auto shape = shapedType.getShape();
      for (size_t i = 0; i < shape.size(); i++) {
        this->dims.push_back(builder.getIndexAttr(shape[i]));
        this->offsets.push_back(builder.getIndexAttr(0));
      }
    } else {
      this->scalar = builder.getIndexAttr(
          attr.getSplatValue<IntegerAttr>().getValue().getSExtValue());
    }
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }
  return success();
}

// parseIntScalar
LogicalResult MaskState::parseIntScalar(Value scalar, const Location &loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  this->scalar = getOpFoldResultOfLayoutInfo(scalar, builder);
  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder))) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder))) {
    return failure();
  }
  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseSub(arith::SubIOp subOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  MaskState lhsState;
  if (failed(lhsState.parse(subOp.getLhs(), loc, builder))) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(subOp.getRhs(), loc, builder))) {
    return failure();
  }
  return this->subStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseDiv(arith::DivSIOp divOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  return failure(); // temporarily disable parseDiv
  MaskState lhsState;
  if (failed(lhsState.parse(divOp.getLhs(), loc, builder))) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(divOp.getRhs(), loc, builder))) {
    return failure();
  }
  return this->divStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)) ||
      !lhsState.isMask()) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)) ||
      !rhsState.isMask()) {
    return failure();
  }

  if (!lhsState.isMask() && !rhsState.isMask()) {
    return failure();
  }

  // Only support both lhs and rhs satisfy `isMask` condition
  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseSel(arith::SelectOp selOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  auto trueValue = selOp.getTrueValue();
  auto falseValue = selOp.getFalseValue();

  MaskState condState;
  auto condition = selOp.getCondition();
  auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp || failed(condState.parse(condition, loc, builder))) {
    return failure();
  }

  MaskState trueState;
  if (failed(trueState.parse(trueValue, loc, builder)) || !trueState.scalar) {
    return failure();
  }

  MaskState falseState;
  if (failed(falseState.parse(falseValue, loc, builder)) ||
      !falseState.scalar) {
    return failure();
  }

  auto trueScalar =
      dyn_cast_if_present<IntegerAttr>(dyn_cast<Attribute>(trueState.scalar));
  auto falseScalar =
      dyn_cast_if_present<IntegerAttr>(dyn_cast<Attribute>(falseState.scalar));

  if (trueScalar && falseScalar) {
    if (trueScalar.getInt() == 1 && falseScalar.getInt() == 0) {
      start = condState.start;
      end = condState.end;
      dims = condState.dims;
      offsets = condState.offsets;
      return success();
    }
  }

  return failure();
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  auto predicate = cmpOp.getPredicate();
  bool isProvenRuntimeExtentUnsignedMask =
      cmpOp->hasAttr(
          mlir::triton::memory_access::IATRuntimeExtentUnsignedMaskTAG) ||
      cmpOp->hasAttr(
          mlir::triton::memory_access::PTSMRuntimeExtentUnsignedMaskTAG);
  bool isTaggedUnsignedUpperBound = isProvenRuntimeExtentUnsignedMask &&
                                    (predicate == arith::CmpIPredicate::ult ||
                                     predicate == arith::CmpIPredicate::ule);
  if (predicate != arith::CmpIPredicate::slt &&
      predicate != arith::CmpIPredicate::sle &&
      predicate != arith::CmpIPredicate::sge &&
      predicate != arith::CmpIPredicate::eq &&
      predicate != arith::CmpIPredicate::ne && !isTaggedUnsignedUpperBound) {
    LLVM_DEBUG({ llvm::dbgs() << "Unsupported cmpi predicate\n"; });
    return failure();
  }

  MaskState lhsState;
  MaskState rhsState;
  auto lhs = cmpOp.getLhs();
  auto rhs = cmpOp.getRhs();

  if (isTaggedUnsignedUpperBound) {
    lhs = unwrapRuntimeExtentUnsignedOperand(lhs, loc, builder);
    rhs = unwrapRuntimeExtentUnsignedOperand(rhs, loc, builder);
    if (!lhs || !rhs)
      return failure();
    predicate = predicate == arith::CmpIPredicate::ult
                    ? arith::CmpIPredicate::slt
                    : arith::CmpIPredicate::sle;
  }

  if (predicate == arith::CmpIPredicate::ne) {
    auto selOp = lhs.getDefiningOp<arith::SelectOp>();
    auto constantOp = rhs.getDefiningOp<arith::ConstantOp>();
    if (!selOp || !constantOp) {
      return failure();
    }
  }

  if (failed(lhsState.parse(lhs, loc, builder))) {
    return failure();
  }

  if (failed(rhsState.parse(rhs, loc, builder))) {
    return failure();
  }

  if (!(!lhsState.scalar && rhsState.scalar)) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "[MaskState] Unsupported cmpi scenario";
    return failure();
  }

  int32_t cmpDim = -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto constDimLength = getConstantIntValue(lhsState.dims[i]);
    if (!constDimLength || constDimLength.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = emitWarning(loc)
                                  << "Unsupported cmpi with more than one  "
                                     "dimension with size larger than 1";
        return failure();
      }
      cmpDim = i;
    }
  }

  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");

  this->offsets = lhsState.offsets;
  this->dims = lhsState.dims;
  switch (predicate) {
  case arith::CmpIPredicate::slt: {
    auto realBound =
        maxOpFoldResult(lhsState.start, rhsState.scalar, loc, builder);
    auto newEnd = minOpFoldResult(lhsState.end, realBound, loc, builder);
    auto newDim = subOpFoldResult(newEnd, lhsState.start, loc, builder);
    auto clampedNewDim = clampToNonNegativeIndex(newDim, loc, builder);

    this->dims[cmpDim] = clampedNewDim;
    break;
  }
  case arith::CmpIPredicate::sle: {
    // lhs <= rhs  <=>  lhs < rhs + 1
    auto rhsPlusOne = addOpFoldResult(rhsState.scalar, builder.getIndexAttr(1),
                                      loc, builder, builder.getIndexType());
    auto realBound = maxOpFoldResult(lhsState.start, rhsPlusOne, loc, builder);
    auto newEnd = minOpFoldResult(lhsState.end, realBound, loc, builder);
    auto newDim = subOpFoldResult(newEnd, lhsState.start, loc, builder);
    auto clampedNewDim = clampToNonNegativeIndex(newDim, loc, builder);

    this->dims[cmpDim] = clampedNewDim;
    break;
  }
  case arith::CmpIPredicate::sge: {
    auto realBound =
        maxOpFoldResult(lhsState.start, rhsState.scalar, loc, builder);
    auto newStart = minOpFoldResult(lhsState.end, realBound, loc, builder);
    auto newOffset = subOpFoldResult(newStart, lhsState.start, loc, builder);
    auto newDim = subOpFoldResult(lhsState.end, newStart, loc, builder);
    auto clampedNewDim = clampToNonNegativeIndex(newDim, loc, builder);

    this->offsets[cmpDim] = newOffset;
    this->dims[cmpDim] = clampedNewDim;
    break;
  }
  case arith::CmpIPredicate::eq: {
    auto newOffset =
        subOpFoldResult(rhsState.scalar, lhsState.start, loc, builder);
    auto newDim = builder.getIndexAttr(1);

    this->offsets[cmpDim] = newOffset;
    this->dims[cmpDim] = newDim;
    break;
  }
  case arith::CmpIPredicate::ne: {
    // only support lhs != 0
    auto rhsScalar = dyn_cast<IntegerAttr>(cast<Attribute>(rhsState.scalar));
    if (!rhsScalar || rhsScalar.getInt() != 0) {
      return failure();
    }

    start = lhsState.start;
    end = lhsState.end;
    break;
  }
  default:
    return failure();
  }
  return success();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location &loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "stride must be 1 for make_range whose result is used "
           "as load or store masks";
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));
  this->offsets.push_back(builder.getIndexAttr(0));
  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location &loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder))) {
    return failure();
  }
  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      this->dims[i] = builder.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }
  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp,
                                    const Location &loc, OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "splat source must be an integer scalar for load/store masks";
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  auto splatAsMask = [&](Operation *userOp) -> bool {
    return TypeSwitch<Operation *, bool>(userOp)
        .Case<arith::AndIOp>([&](arith::AndIOp andOp) { return true; })
        .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
          return selectOp.getCondition() == dst;
        })
        .Case<triton::LoadOp>(
            [&](triton::LoadOp loadOp) { return loadOp.getMask() == dst; })
        .Case<triton::StoreOp>(
            [&](triton::StoreOp storeOp) { return storeOp.getMask() == dst; })
        .Case<triton::AtomicRMWOp>([&](triton::AtomicRMWOp atomicOp) {
          return atomicOp.getMask() == dst;
        })
        .Default([&](Operation *op) { return false; });
  };

  if (src.getType().isInteger(1) && !splatOp->use_empty() &&
      llvm::all_of(splatOp->getUsers(), splatAsMask)) {
    for (auto s : dstShape) {
      auto currentDim = mulOpFoldResult(builder.getIndexAttr(s), this->scalar,
                                        loc, builder, builder.getIndexType());
      this->dims.push_back(currentDim);
      this->offsets.push_back(builder.getIndexAttr(0));
    }

    this->scalar = nullptr;
    return success();
  }

  for (auto s : dstShape) {
    this->dims.push_back(builder.getIndexAttr(s));
    this->offsets.push_back(builder.getIndexAttr(0));
  }
  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location &loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder))) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "Expect changed dimention to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));
  this->offsets.insert(this->offsets.begin() + axis, builder.getIndexAttr(0));

  return success();
}

LogicalResult MaskState::parseInsert(tensor::InsertOp insertOp,
                                     const Location &loc, OpBuilder &builder) {
  if (auto tensorType = dyn_cast<RankedTensorType>(insertOp.getType())) {
    if (llvm::all_of(tensorType.getShape(),
                     [](int64_t dim) { return dim == 1; })) {
      SmallVector<Value> indices(
          tensorType.getRank(), builder.create<arith::ConstantIndexOp>(loc, 0));
      auto scalarlizeTensor =
          builder.create<tensor::ExtractOp>(loc, insertOp, indices).getResult();
      this->offsets = SmallVector<OpFoldResult>(tensorType.getRank(),
                                                builder.getIndexAttr(0));
      this->dims = SmallVector<OpFoldResult>(tensorType.getRank(),
                                             builder.getIndexAttr(1));
      this->scalar = getOpFoldResultOfLayoutInfo(scalarlizeTensor, builder);
      return success();
    }
  }
  return failure();
}

void MaskState::eraseInsertedOps(Operation *rawOp, PatternRewriter &rewriter) {
  auto moduleOp = rawOp->getParentOfType<ModuleOp>();
  SmallVector<Operation *> worklist;
  moduleOp->walk([&](Operation *op) {
    if (isOpTriviallyDead(op) && op->use_empty()) {
      worklist.push_back(op);
    }
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op) || !op->use_empty()) {
      continue;
    }
    if (!op->getBlock()) {
      continue;
    }
    SmallVector<Operation *> operandDefOps;
    for (Value value : op->getOperands()) {
      if (auto defOp = value.getDefiningOp()) {
        if (defOp->getBlock()) {
          operandDefOps.push_back(defOp);
        }
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[MaskState]==> inserted op: \n"
                   << *op << "\n[MaskState]<== is removed\n";
    });

    rewriter.eraseOp(op);
    for (auto defOp : operandDefOps) {
      worklist.push_back(defOp);
    }
  }
}

std::optional<MaskState> runMaskAnalysis(Operation *op, OpBuilder &builder) {
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    return runMaskAnalysisImpl(loadOp, builder);
  }
  if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    return runMaskAnalysisImpl(storeOp, builder);
  }
  if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
    return runMaskAnalysisImpl(atomicRMWOp, builder);
  }
  return std::nullopt;
}

} // namespace triton

} // namespace mlir
