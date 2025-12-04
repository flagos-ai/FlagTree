//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/PtrAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <set>

#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

namespace triton {

static void assertValidUnrealizedCast(UnrealizedConversionCastOp op) {
  assert(op && op->hasAttr(ModuloState::WraparoundAttr) &&
         op.getInputs().size() == 3 &&
         op.getInputs()[0].getDefiningOp<memref::ReinterpretCastOp>() &&
         op.getInputs()[1].getDefiningOp<memref::ReinterpretCastOp>() &&
         op.getInputs()[2].getDefiningOp<triton::AddPtrOp>());
}

MemRefType PtrState::getResultMemrefType(MLIRContext *context, int64_t offset,
                                         ArrayRef<int64_t> resultShape,
                                         bool useDynamicStrides) const {

  SmallVector<int64_t> staticStrides;
  if (useDynamicStrides) {
    staticStrides.append(strides.size(), ShapedType::kDynamic);
  } else {
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  }

  auto elementType = cast<BaseMemRefType>(source.getType()).getElementType();
  auto layout =
      StridedLayoutAttr::get(source.getContext(), offset, staticStrides);

  return MemRefType::get(resultShape, elementType, layout);
}

OpFoldResult
PtrState::accumulateTargetOffset(Location loc,
                                 ConversionPatternRewriter &rewriter) const {
  OpFoldResult targetOffset = rewriter.getIndexAttr(0);
  for (auto o : offsets) {
    targetOffset = addOFRs(targetOffset, o, loc, rewriter);
  }
  return targetOffset;
}

int64_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         modulos.size() == offsets.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::hasModulo() const {
  return llvm::any_of(modulos, [](auto mod) { return mod.has_value(); });
}

void PtrState::addState(const PtrState &lhsState, const PtrState &rhsState,
                        Location loc, ConversionPatternRewriter &rewriter) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  // at most one of lhs and rhs should have valid source, since otherwise we
  // will be losing information
  assert(!(lhsState.source && rhsState.source));
  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        rewriter.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  for (uint64_t i = 0; i < lhsState.sizes.size(); i++) {
    auto newOffset =
        addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, rewriter);
    offsets.push_back(newOffset);

    auto newStride =
        addOFRs(lhsState.strides[i], rhsState.strides[i], loc, rewriter);
    strides.push_back(newStride);

    sizes.push_back(lhsState.sizes[i]);

    assert(!lhsState.hasModulo() ||
           !rhsState.hasModulo() && "AddPtr where both lhs and rhs containing "
                                    "modulo operators not supported");

    modulos.push_back(lhsState.modulos[i].has_value() ? lhsState.modulos[i]
                                                      : rhsState.modulos[i]);
  }
}

void PtrState::mulState(const PtrState &lhsState, const PtrState &rhsState,
                        const Location loc,
                        ConversionPatternRewriter &rewriter) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  assert(!(lhsState.source && rhsState.source));

  assert((lhsState.scalar || rhsState.scalar) &&
         !(lhsState.scalar && rhsState.scalar) &&
         "currently does not support both tensors are effectively non-scalar");

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  for (uint64_t i = 0; i < lhs->sizes.size(); i++) {
    OpFoldResult newOffset =
        mulOFRValue(lhs->offsets[i], rhs->scalar, loc, rewriter);
    OpFoldResult newStride =
        mulOFRValue(lhs->strides[i], rhs->scalar, loc, rewriter);
    offsets.push_back(newOffset);
    strides.push_back(newStride);
    sizes.push_back(lhs->sizes[i]);
  }

  assert(llvm::all_of(rhsState.modulos,
                      [](auto rhs) { return !rhs.has_value(); }));

  modulos = lhs->modulos;
}

SmallVector<memref::ReinterpretCastOp>
PtrState::createStackedCastOps(ArrayRef<int64_t> resultShape,
                               const Location loc,
                               ConversionPatternRewriter &rewriter) const {

  assert(resultShape.size() == 2);
  assert(getRank() == 2);
  assert(modulos[0].has_value() && !modulos[1].has_value());

  Value targetOffset =
      ofrToIndexValue(accumulateTargetOffset(loc, rewriter), loc, rewriter);

  //////////////////////////////////////////////////////////////////////////////
  //
  // Handling stacked wraparound
  //
  // We do not support cases where the target offset has already overflown the
  // number of rows. See side-by-side wraparound for details.
  //
  //////////////////////////////////////////////////////////////////////////////
  //    We're loading a tensor of dim (rowSize, colSize)
  //    d1 + d2 = rowSize
  //    d2 is the number of rows that overflow
  //
  //                       cols
  //
  //               wrappedAroundOff
  //      --------------*------------*--------
  //      |        d2   |            |       |
  //      |             |------------|       |
  //  rows|                                  |
  //      |                                  |
  //      |           targetOffset           |
  //      |             *------------|       |
  //      |             |            |       |
  //      |         d1  |            |       |
  //      |             | clampedOff |       |
  //      --------------*---------------------
  //                    |  overflow  |
  //                    *-------------
  //                 nextOff
  //
  //    wrappedAroundOff = targetOffset % cols
  //    clampedOff = (rows * strideRows) + wrappedAroundOff
  //
  //          clampedOff - targetOffset
  //    d1 = --------------------
  //              strideRows

  auto resultType = getResultMemrefType(
      rewriter.getContext(), /* offset */ ShapedType::kDynamic,
      /* result shape */
      SmallVector<int64_t>{
          ShapedType::kDynamic, // Row is dynamic, in most cases, this should be
                                // the same as the original row. The last chunk
                                // may be smaller due to wrapping around.
          resultShape[1],       // Col stays the same.
      },
      true /*useDynamicStrides*/);

  Value rowSize = ofrToIndexValue(sizes[0], loc, rewriter);
  Value colSize = ofrToIndexValue(sizes[1], loc, rewriter);

  Value strideRow = ofrToIndexValue(strides[0], loc, rewriter);
  Value strideCol = ofrToIndexValue(strides[1], loc, rewriter);

  Value modRow = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), modulos[0]->size);

  // First chunk
  Value wrappedAroundOff =
      rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
  Value clampedOff = rewriter.create<arith::MulIOp>(loc, modRow, strideRow);
  clampedOff =
      rewriter.create<arith::AddIOp>(loc, clampedOff, wrappedAroundOff);
  Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
  d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);

  SmallVector<Value> sizes1{d1, colSize};
  memref::ReinterpretCastOp cast1 = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, source, targetOffset, sizes1,
      ValueRange{strideRow, strideCol});

  // Second chunk
  Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);
  SmallVector<Value> sizes2{d2, colSize};
  memref::ReinterpretCastOp cast2 = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, source, wrappedAroundOff, sizes2,
      ValueRange{strideRow, strideCol});

  return {cast1, cast2};
}

SmallVector<memref::ReinterpretCastOp>
PtrState::createSideBySideCastOps(ArrayRef<int64_t> resultShape,
                                  const Location loc,
                                  ConversionPatternRewriter &rewriter) const {

  assert(resultShape.size() == 2);
  assert(getRank() == 2 && !modulos[0].has_value() && modulos[1].has_value());

  // Accumulate final offset
  Value targetOffset =
      ofrToIndexValue(accumulateTargetOffset(loc, rewriter), loc, rewriter);

  //////////////////////////////////////////////////////////////////////////////
  //
  // Handling side-by-side wraparound
  //
  // Note: We do not support cases where the target has already overflown the
  // number of columns! This is because in PtrAnalysis, the offset has already
  // been collapsed into a single dimension, so it is ambiguous to determine
  // whether the offset actually overflows or just refers to an element on the
  // subsequent rows.
  //
  // Same limitations apply to the stacked wraparound case.
  //
  //////////////////////////////////////////////////////////////////////////////
  //
  //    nextOffset - targetOffset = colSize
  //    d1 + d2 = colSize
  //                          N
  //                                x            clampedOffset
  //      --------------------------*----------------*-----*
  //      |                                          |     nextOffset (might
  //      |                    targetOffset          |             overflow)
  //  y   *-----                    *----------------|
  //      |    |                    |                |
  //  M   |-----                    -----------------|
  //      | d2                              d1       |
  //      --------------------------------------------
  //
  //    x = targetOffset % N
  //    nextOffset = x + colSize
  //    clampedOffset = min(nextOffset, N)
  //    d1 = clampedOffset - x
  //
  //////////////////////////////////////////////////////////////////////////////

  SmallVector<memref::ReinterpretCastOp> casts;

  auto resultType = getResultMemrefType(
      rewriter.getContext(), /* offset */ ShapedType::kDynamic,
      /* result shape */
      SmallVector<int64_t>{
          resultShape[0],      // Row stays the same
          ShapedType::kDynamic // Column is dynamic, in most cases, this should
                               // be the same as the original column. The last
                               // chunk may be smaller due to wrapping around.
      },
      true /*useDynamicStrides*/);

  Value rowSize = ofrToIndexValue(sizes[0], loc, rewriter);
  Value colSize = ofrToIndexValue(sizes[1], loc, rewriter);

  Value modN = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   modulos[1]->size);

  Value x = rewriter.create<arith::RemSIOp>(loc, targetOffset, modN);
  Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, x);

  SmallVector<Value> strideVals = ofrsToIndexValues(strides, loc, rewriter);

  // First chunk
  Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
  Value clampedOffset = rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
  Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);
  SmallVector<Value> sizes1{rowSize, d1};

  auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, source, targetOffset, sizes1, strideVals);

  // Second chunk
  Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);
  SmallVector<Value> sizes2{rowSize, d2};

  auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, source, y, sizes2, strideVals);

  return {cast1, cast2};
}

memref::ReinterpretCastOp
PtrState::createCastOp(ArrayRef<int64_t> resultShape, const Location loc,
                       ConversionPatternRewriter &rewriter) const {
  // Accumulate final offset
  OpFoldResult targetOffset = accumulateTargetOffset(loc, rewriter);

  // Create result MemRefType
  SmallVector<int64_t> staticOffset;
  SmallVector<Value> dynamicOffset;
  dispatchIndexOpFoldResult(targetOffset, dynamicOffset, staticOffset);

  auto resultType =
      getResultMemrefType(rewriter.getContext(), staticOffset[0], resultShape);

  // Create reinterpret cast
  return rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, source, targetOffset, sizes, strides);
}

void PtrAnalysis::visitOperandAdd(
    arith::AddIOp addOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(addOp.getLhs(), lhsState, loc, rewriter, knownPtrs);

  PtrState rhsState;
  visitOperand(addOp.getRhs(), rhsState, loc, rewriter, knownPtrs);

  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    assert(0 && "Current do not support this pattern: a + arange(0, K) % M");
  }

  state.addState(lhsState, rhsState, loc, rewriter);
}

void PtrAnalysis::visitOperandMul(
    arith::MulIOp mulOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(mulOp.getLhs(), lhsState, loc, rewriter, knownPtrs);

  PtrState rhsState;
  visitOperand(mulOp.getRhs(), rhsState, loc, rewriter, knownPtrs);

  state.mulState(lhsState, rhsState, loc, rewriter);
}

void PtrAnalysis::visitOperandRem(
    arith::RemSIOp remOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  PtrState rhsState;
  visitOperand(remOp.getRhs(), rhsState, loc, rewriter, knownPtrs);
  assert(rhsState.scalar);

  visitOperand(remOp.getLhs(), state, loc, rewriter, knownPtrs);

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  assert(llvm::all_of(state.modulos,
                      [](auto modState) { return !modState.has_value(); }) &&
         "No support for multiple modulo within an expression");

  if (state.getRank() == 1) {
    // Apply the modulo before expanding shape, the common pattern is
    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
    // stride_ak)
    state.modulos.back() = ModuloState{rhsState.scalar};
  } else if (state.getRank() == 2) {
    // torch inductor expands the tensor shape before applying the modulo.
    //
    // We only support either:
    // - (tl.arange(0, end)[:, None] % mod), or
    // - (tl.arange(0, end)[None, :] % mod)
    //
    // In both cases, we apply the modulo to the non-singleton dimension.
    auto shape = cast<TensorType>(remOp.getResult().getType()).getShape();
    if (shape[0] == 1) {
      state.modulos[1] = ModuloState{rhsState.scalar};
    } else if (shape[1] == 1) {
      state.modulos[0] = ModuloState{rhsState.scalar};
    } else {
      assert(false && "Taking modulo on a 2D tensor with no singleton "
                      "dimension not supported");
    }
  } else {
    assert(false && "Unsupported modulo pattern");
  }
}

void PtrAnalysis::visitOperandMakeRange(
    triton::MakeRangeOp rangeOp, PtrState &state, Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.offsets.push_back(rewriter.getIndexAttr(start));
  state.sizes.push_back(rewriter.getIndexAttr(shape[0]));
  state.strides.push_back(rewriter.getIndexAttr(stride));
  state.modulos.push_back(std::nullopt);
}

void PtrAnalysis::visitOperandExpandDims(
    triton::ExpandDimsOp expandDimsOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  // `getSrc` now returns a TypedValue of RankedTensorType. We modify these
  // operands in-place and turn them into memrefs in loops, so we have to bypass
  // the cast by using getSrcMutable. These are temporary fix only since
  // we will be moving over to StructuredPtrAnalysis soon which separate out the
  // memref conversion.
  visitOperand(expandDimsOp.getSrcMutable().get(), state, loc, rewriter,
               knownPtrs);

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  state.offsets.insert(state.offsets.begin() + axis, rewriter.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, rewriter.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, rewriter.getIndexAttr(0));
  state.modulos.insert(state.modulos.begin() + axis, std::nullopt);
}

void PtrAnalysis::visitOperandBroadcast(
    triton::BroadcastOp broadcastOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  // `getSrc` now returns a TypedValue of RankedTensorType. We modify these
  // operands in-place and turn them into memrefs in loops, so we have to bypass
  // the cast by using getSrcMutable. These are temporary fix only since
  // we will be moving over to StructuredPtrAnalysis soon which separate out the
  // memref conversion.
  auto src = broadcastOp.getSrcMutable().get();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  visitOperand(src, state, loc, rewriter, knownPtrs);

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      state.sizes[i] = rewriter.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }
}

void PtrAnalysis::visitOperandSplat(
    triton::SplatOp splatOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  visitOperand(src, state, loc, rewriter, knownPtrs);

  if (isa<IntegerType, triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.offsets.push_back(rewriter.getIndexAttr(0));
      state.sizes.push_back(rewriter.getIndexAttr(s));
      state.strides.push_back(rewriter.getIndexAttr(0));
      state.modulos.push_back(std::nullopt);
    }
  } else {
    // src is a memref that represent a scalar pointer; it should have
    // one dimension of size 1. This happens inside a for loop that
    // originally has an init arg that is a tensor of pointers; this arg
    // would have been replaced by rewriteForOp.
    auto srcType = cast<MemRefType>(src.getType());
    assert(srcType.getRank() == 1 && state.getRank() == 1 &&
           "splat MemRef source should have rank 1");
    assert(srcType.getShape()[0] == 1 &&
           getIntAttr(state.sizes[0]).value() == 1 &&
           "splat MemRef source should have size 1");

    // Stride[0] will have value of 1 set in visitOperandAddPtr. This
    // value will be represented by a constOp. Clear this value.
    state.strides[0] = rewriter.getIndexAttr(0);

    for (auto [i, s] : llvm::enumerate(dstShape)) {
      if (i == 0) {
        state.sizes[i] = rewriter.getIndexAttr(s);
        continue;
      }
      state.offsets.push_back(rewriter.getIndexAttr(0));
      state.sizes.push_back(rewriter.getIndexAttr(s));
      state.strides.push_back(rewriter.getIndexAttr(0));
      state.modulos.push_back(std::nullopt);
    }
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar)
    state.offsets[0] = state.scalar;
}

void PtrAnalysis::visitOperandMakeTensorPtr(
    triton::MakeTensorPtrOp makeTensorPtrOp, PtrState &state,
    const Location loc, ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  auto remappedValue = rewriter.getRemappedValue(makeTensorPtrOp);
  if (auto castOp = remappedValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    visitOperandReintCast(castOp, state, loc, rewriter, knownPtrs);
  } else {
    llvm_unreachable("Expect value to me mapped to a memref.reinterpret_cast");
  }
}

void PtrAnalysis::visitOperandAddptr(
    triton::AddPtrOp addptrOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  PtrState ptrState;
  visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), rewriter,
               knownPtrs);

  PtrState offsetState;
  visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(), rewriter,
               knownPtrs);

  assert(ptrState.source && "ptr field should provide source / base pointer");

  // Handle the special case when we are in a for loop, ptr is originally a
  // scalar pointer but replaced with a memref. In this case, ptrState will have
  // rank 1 and offsetState will have rank 0.
  // TODO:
  //  Passing a block argument pointer directly into a for loop not supported
  if (ptrState.getRank() == 1 && offsetState.getRank() == 0) {
    offsetState.sizes.push_back(rewriter.getIndexAttr(1));
    offsetState.offsets.push_back(offsetState.scalar);
    offsetState.strides.push_back(rewriter.getIndexAttr(0));
    offsetState.modulos.push_back(std::nullopt);
  }

  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  state.addState(ptrState, offsetState, addptrOp.getLoc(), rewriter);
}

void PtrAnalysis::visitOperandReintCast(
    memref::ReinterpretCastOp reintCastOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  state.offsets = reintCastOp.getMixedOffsets();
  state.sizes = reintCastOp.getMixedSizes();
  state.strides = reintCastOp.getMixedStrides();
  state.source = reintCastOp.getSource();
  state.modulos.append(state.sizes.size(), std::nullopt);

  // getMixedOffsets produces staticOffsets (which is the result of collapsing
  // multiple dimensions). Populate the rest of the dimensions with zeroes.
  assert(state.offsets.size() == 1);
  for (size_t i = 1; i < state.sizes.size(); i++) {
    state.offsets.push_back(rewriter.getIndexAttr(0));
  }

  // Regular Triton programs cannot express patterns of size 1 and non-zero
  // stride; we only set it that way to make memrefs work. Set stride back to
  // zero if this scenario detected.
  for (size_t i = 0; i < state.strides.size(); i++) {
    auto strideIntAttr = getIntAttr(state.strides[i]);
    auto sizeIntAttr = getIntAttr(state.sizes[i]);

    assert(sizeIntAttr);
    if (sizeIntAttr.value() == 1 && strideIntAttr) {
      state.strides[i] = rewriter.getIndexAttr(0);
    }
  }
}

void PtrAnalysis::visitOperand(
    Value operand, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {

  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return;
  }

  if (isa<IntegerType>(operand.getType())) {
    auto castOp = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return;
  }

  if (isa<triton::PointerType>(operand.getType())) {
    auto remappedPtr = rewriter.getRemappedValue(operand);
    assert(remappedPtr);

    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc, rewriter,
                           knownPtrs);
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        visitOperandMakeTensorPtr(makeTensorOp, state, loc, rewriter,
                                  knownPtrs);
      } else {
        llvm_unreachable("Unexpected operand defining operation");
      }
    } else {
      state.source = remappedPtr;
    }
    return;
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    visitOperandAdd(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    visitOperandMul(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    visitOperandMakeRange(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    visitOperandBroadcast(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    visitOperandSplat(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    visitOperandExpandDims(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    visitOperandAddptr(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    visitOperandConstSplat(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    visitOperandRem(op, state, loc, rewriter, knownPtrs);
  } else {
    operand.dump();
    llvm_unreachable("encountered addptr operand produced by an "
                     "unsupported operation");
  }
}

void PtrAnalysis::visitOperandConstSplat(
    arith::ConstantOp op, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = rewriter.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(rewriter, constAttr,
                                                rewriter.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(rewriter.getIndexAttr(0));
    }

    state.sizes.push_back(rewriter.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(rewriter.getIndexAttr(0));
    state.modulos.push_back(std::nullopt);
  }
}

void PtrAnalysis::rewriteAddptrOp(
    triton::AddPtrOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  // any inserted instruction should be before this addptr
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  PtrState state;
  visitOperandAddptr(op, state, op.getLoc(), rewriter, knownPtrs);

  // If the result is a scalar pointer, visitOperandAddptr will not populate
  // sizes, strides, and offsets. We need to do it here.
  if (state.sizes.size() == 0) {
    state.sizes.push_back(rewriter.getIndexAttr(1));
    state.strides.push_back(rewriter.getIndexAttr(0));
    state.offsets.push_back(state.scalar);
    state.modulos.push_back(std::nullopt);
  }

  SmallVector<int64_t> scalarShape(1, 1);
  ArrayRef<int64_t> resultShape;
  if (auto shapedType = dyn_cast<ShapedType>(op.getResult().getType())) {
    resultShape = shapedType.getShape();
  } else {
    // scalar pointer, should produce a one dimensional memref
    resultShape = scalarShape;
    assert(state.getRank() == 1);
  }

  knownPtrs[op.getResult()] = state;

  // If there are dimensions with size 1 and stride 0, replace 0 stride with the
  // product of sizes of all lower dimensions. This avoids creating memref with
  // zero stride. Note that we store the unmodified state into knownPtrs, since
  // any following pointer arithmetic operations should use the original 0
  // stride.
  auto accum_size = 1;
  for (int i = state.sizes.size() - 1; i >= 0; i--) {
    auto strideIntAttr = getIntAttr(state.strides[i]);
    auto sizeIntAttr = getIntAttr(state.sizes[i]);

    assert(sizeIntAttr);
    if (sizeIntAttr.value() == 1 && strideIntAttr && strideIntAttr.value() == 0)
      state.strides[i] = rewriter.getIndexAttr(accum_size);

    accum_size *= sizeIntAttr.value();
  }

  Value src;

  if (llvm::any_of(state.modulos, [](auto mod) { return mod.has_value(); })) {
    assert(state.modulos.size() == 2);
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    SmallVector<memref::ReinterpretCastOp> casts;
    StringRef type;

    if (!state.modulos[0].has_value() && state.modulos[1].has_value()) {
      casts = state.createSideBySideCastOps(resultShape, op.getLoc(), rewriter);
      type = ModuloState::WraparoundSideBySide;
    } else if (state.modulos[0].has_value() && !state.modulos[1].has_value()) {
      casts = state.createStackedCastOps(resultShape, op.getLoc(), rewriter);
      type = ModuloState::WraparoundStacked;
    } else {
      assert(false && "not supported");
    }

    auto resultType = state.getResultMemrefType(
        rewriter.getContext(), ShapedType::kDynamic, resultShape);

    UnrealizedConversionCastOp combinedCast =
        rewriter.create<UnrealizedConversionCastOp>(
            op.getLoc(), resultType,
            ValueRange{casts[0].getResult(), casts[1].getResult(),
                       op.getResult()});

    combinedCast->setAttr(ModuloState::WraparoundAttr,
                          rewriter.getStringAttr(type));

    src = combinedCast.getResult(0);

    LLVM_DEBUG({
      llvm::dbgs() << "combine cast for split pointers:\n";
      combinedCast.getOperation()->print(
          llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";
    });

  } else {
    memref::ReinterpretCastOp castOp =
        state.createCastOp(resultShape, op.getLoc(), rewriter);

    src = castOp.getResult();

    LLVM_DEBUG({
      llvm::dbgs() << "cast MemRefType:\n";
      castOp.getOperation()->print(llvm::dbgs(),
                                   OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";
    });
  }

  state.source = src;
  rewriter.replaceOp(op, src);
  rewriter.restoreInsertionPoint(origIp);
}

void PtrAnalysis::rewriteAdvanceOp(
    triton::AdvanceOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  OpBuilder::InsertionGuard insertionGuard{rewriter};
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();

  PtrState ptrState;
  visitOperand(op.getOperand(0), ptrState, loc, rewriter, knownPtrs);

  auto incrementOffsets = op.getOffsets();

  SmallVector<Value> newOffsets;
  for (auto [increment, offset, stride] :
       llvm::zip(incrementOffsets, ptrState.offsets, ptrState.strides)) {
    Value offsetValue;
    if (auto offsetIntAttr = getIntAttr(offset)) {
      auto constOp = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(0));
      offsetValue = constOp.getResult();
    } else {
      offsetValue = cast<Value>(offset);
    }
    auto castOp = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), increment);
    auto mulOp = rewriter.create<arith::MulIOp>(loc, castOp.getResult(),
                                                cast<Value>(stride));
    auto addOp =
        rewriter.create<arith::AddIOp>(loc, mulOp.getResult(), offsetValue);
    newOffsets.push_back(addOp.getResult());
  }

  ptrState.offsets.clear();

  for (auto offset : newOffsets) {
    ptrState.offsets.push_back(offset);
  }

  SmallVector<int64_t> scalarShape(1, 1);
  ArrayRef<int64_t> resultShape;
  auto pointerType = cast<mlir::triton::PointerType>(op.getResult().getType());
  if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
    resultShape = shapedType.getShape();
  } else {
    // scalar pointer, should produce a one dimensional memref
    resultShape = scalarShape;
    assert(ptrState.getRank() == 1);
  }

  auto newOp = ptrState.createCastOp(resultShape, loc, rewriter);

  rewriter.replaceOp(op, newOp.getResult());

  knownPtrs[newOp.getResult()] = ptrState;
}

void PtrAnalysis::rewriteYieldOp(
    scf::YieldOp op, ConversionPatternRewriter &rewriter,
    const IndexMapSet &levelToBlockArgIndex, const int level,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  // any inserted instruction should be before this yield
  OpBuilder::InsertionGuard insertionGuard{rewriter};
  rewriter.setInsertionPoint(op);

  auto adaptor = scf::YieldOp::Adaptor(op);

  SmallVector<PtrState, 5> initArgState;
  SmallVector<Value> operands(adaptor.getOperands());
  // Track the second chunks of modulo pointers so that we can append them to
  // the yield results
  SmallVector<Value> moduloSecondChunks;

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // PtrState for those values.
  for (auto [i, v] : llvm::enumerate(adaptor.getOperands())) {
    if (auto mappedV = rewriter.getRemappedValue(v)) {
      // If this value is a tensor of pointers produced by AddPtrOp,
      // we should have already converted to a ReinterpretCastOp without
      // layout information for the normal cases, or to an
      // UnrealizedConversionCastOp for the split pointer case.
      if (v.getDefiningOp<triton::AddPtrOp>() ||
          v.getDefiningOp<triton::AdvanceOp>() ||
          v.getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (auto castOp = mappedV.getDefiningOp<UnrealizedConversionCastOp>()) {
          assertValidUnrealizedCast(castOp);
          auto castInputs = castOp.getInputs();
          v = castOp.getResult(0);
          operands[i] = castInputs[0];
          moduloSecondChunks.push_back(castInputs[1]);
        } else if (auto castOp =
                       mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
          v = castOp;
        } else {
          llvm_unreachable("mapped value defined by an unexpected op");
        }
      } else {
        // If this value is not a tensor of pointers, we will use the
        // mapped value, and rely on the conversion will happen later
        // automatically when we legalize loop body.

        // TODO:
        //  The scenario where a value is a tensor of pointers but not
        //  produced by AddPtrOp is not supported
        if (isa<TensorType>(mappedV.getType()) &&
            isa<triton::PointerType>(
                dyn_cast<TensorType>(mappedV.getType()).getElementType()))
          llvm_unreachable("unsupported scenario where a value is a tensor of "
                           "pointers but not produced by AddPtrOp");
        v = mappedV;
      }
    }

    if (levelToBlockArgIndex.find(level) == levelToBlockArgIndex.end())
      continue;
    auto thisSet = levelToBlockArgIndex.find(level)->second;
    if (thisSet.find(i) == thisSet.end())
      continue;

    auto reintCastOp = v.getDefiningOp<memref::ReinterpretCastOp>();
    auto unrealizedCastOp = v.getDefiningOp<UnrealizedConversionCastOp>();

    assert(
        reintCastOp ||
        (unrealizedCastOp &&
         unrealizedCastOp->hasAttr(ModuloState::WraparoundAttr)) ||
        (isa<TensorType>(v.getType()) &&
         isa<IndexType>(dyn_cast<TensorType>(v.getType()).getElementType())));

    PtrState state;
    if (reintCastOp) {
      visitOperandReintCast(reintCastOp, state, op.getLoc(), rewriter,
                            knownPtrs);
    } else if (unrealizedCastOp) {
      assertValidUnrealizedCast(unrealizedCastOp);
      visitOperandUnrealizedCast(unrealizedCastOp, state, op.getLoc(), rewriter,
                                 knownPtrs);
    } else {
      visitOperand(v, state, op.getLoc(), rewriter, knownPtrs);
    }
    initArgState.push_back(state);
  }

  // For each of the PtrState recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for (auto s : state.offsets) {
      // offsets can be IntAttr zeroes, since reinterpret_cast collapses
      // them for the input memref, and the for loop may not update
      // offsets other than offsets[0]. Create constants Values for those
      // zeroes.
      if (auto sIntAttr = getIntAttr(s)) {
        assert(sIntAttr.value() == 0 && "attribute offsets should be zeroes");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(cast<Value>(s));
      }
    }

    for (auto s : state.strides) {
      assert(!getIntAttr(s) && "PtrState strides for yield within for "
                               "loop not expected to be "
                               "attribute.");
      operands.push_back(cast<Value>(s));
    }
  }

  for (auto chunk : moduloSecondChunks) {
    operands.push_back(chunk);
  }

  // Yield is a terminator op that must be at the end of the function
  rewriter.setInsertionPointAfter(op);
  auto newOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
  assert(op->getNumResults() == 0);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

// From an unrealized_conversion_cast which takes in two reinterpret_casts
// representing two chunks, we need to get back the full pointer state. We
// cannot rebuild the original state from the two reinterpret_casts similarly to
// the normal case. To solve this, we attach the original addptr as the third
// operand to the unrealized_cast so that we can manually rebuild the state.
void PtrAnalysis::visitOperandUnrealizedCast(
    UnrealizedConversionCastOp op, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assertValidUnrealizedCast(op);

  auto origPtr = op.getInputs()[2];
  if (knownPtrs.contains(origPtr)) {
    state = knownPtrs.at(origPtr);
  } else {
    visitOperandAddptr(origPtr.getDefiningOp<triton::AddPtrOp>(), state, loc,
                       rewriter, knownPtrs);
  }
}

struct ModuloChunkInitArg {
  Value reinterpretCast = nullptr;
  // where in the init args is the first chunk placed
  size_t initArgIndex = -1;
};

void PtrAnalysis::rewriteForOp(
    scf::ForOp op, ConversionPatternRewriter &rewriter,
    IndexMapSet &levelToBlockArgIndex, const int level,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  SmallVector<Value> newInitArgs;

  SmallVector<std::pair<int, PtrState>, 5> initArgIndexState;
  SmallVector<std::pair<int, PtrState>, 5> knownPtrsTmp;

  // If we have a load op that uses a modulo pointer, we need to insert both of
  // the memref chunks to the init args. We reuse the sizes from the original
  // memrefs. This data structure keeps track of where these additional init
  // args should be inserted.
  //
  // As an example, if we have a 2D memrefs being split, we first put the first
  // chunk in the order as it appears. Then, once all of the original init args
  // are processed, we insert their offsets and strides, and finally the second
  // chunk.
  SmallVector<std::tuple<UnrealizedConversionCastOp,
                         SmallVector<ModuloChunkInitArg>, PtrState>,
              6>
      moduloStates;

  // Amongst the init args, track the indices that map to the first chunk of a
  // modulo pair. This is used to distinguish between the normal
  // reinterpret_casts whose return types need to be rewritten to match what the
  // for loop is yielding.
  DenseSet<size_t> moduloInitArgIndices;

  // Create a new list of init args
  for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
    auto mappedV = rewriter.getRemappedValue(arg);
    memref::ReinterpretCastOp reintCastOp;
    UnrealizedConversionCastOp unrealizedCastOp;

    // If this init arg is supposed to be remapped, use the remapped
    // value instead. In addition, if this init arg is a memref created
    // by a reinterpret_cast or a tensor of index, there is a chance that
    // it will be used in addptr. Create PtrState for each such init arg.
    if (mappedV) {
      // TODO:
      //  Passing a block argument pointer directly into a for loop not
      //  supported.
      assert(!(dyn_cast<BlockArgument>(mappedV) &&
               isa<UnrankedMemRefType>(mappedV.getType())) &&
             "cannot take pointer block argument as init arg for for loop");
      if (auto op = mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
        reintCastOp = op;
        newInitArgs.push_back(mappedV);
      } else if (auto op =
                     mappedV.getDefiningOp<UnrealizedConversionCastOp>()) {
        assertValidUnrealizedCast(op);
        unrealizedCastOp = op;
        auto inputs = unrealizedCastOp.getInputs();

        SmallVector<ModuloChunkInitArg> initArgData{
            ModuloChunkInitArg{inputs[0], i},
            ModuloChunkInitArg{inputs[1]},
        };

        moduloInitArgIndices.insert(i);
        moduloStates.push_back(
            std::make_tuple(unrealizedCastOp, initArgData, PtrState{}));

        newInitArgs.push_back(inputs[0]);
      } else {
        newInitArgs.push_back(mappedV);
      }

    } else {
      newInitArgs.push_back(arg);
    }

    auto indexTensor =
        isa<TensorType>(arg.getType()) &&
        isa<IndexType>(dyn_cast<TensorType>(arg.getType()).getElementType());

    if (!unrealizedCastOp && !reintCastOp && !indexTensor)
      continue;

    PtrState state;
    if (reintCastOp) {
      visitOperandReintCast(reintCastOp, state, op.getLoc(), rewriter,
                            llvm::SmallDenseMap<Value, PtrState>(0));
    } else if (unrealizedCastOp) {
      visitOperandUnrealizedCast(unrealizedCastOp, state, op.getLoc(), rewriter,
                                 llvm::SmallDenseMap<Value, PtrState>(0));
      std::get<2>(moduloStates.back()) = state;
    } else {
      visitOperand(arg, state, op.getLoc(), rewriter,
                   llvm::SmallDenseMap<Value, PtrState>(0));
    }

    // Record the PtrState for later processing
    initArgIndexState.push_back(std::make_pair(i, state));
  }

  // Set insertion point to be before the for loop for new variables passed
  // into the new loop.
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  // For each of the PtrState recorded in the last step, insert new
  // instructions to describe offset and stride for each dimension and append
  // them to init args
  for (auto [i, state] : initArgIndexState) {
    // For each dimension, if the corresponding offset and stride is an
    // integer attribute, create a constant value and append them at the
    // end of init arg list.
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.offsets[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(cast<Value>(s));
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.strides[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(cast<Value>(s));
      }
    }

    // Note that we want the knownPtrs to be indexed by block arg, but we
    // only have index for now. Also, the state we record is the init
    // arg, but want to to use newly created block arg. These block args
    // are not created yet. We will translate this mapping later.
    knownPtrsTmp.push_back(std::make_pair(i, state));
    levelToBlockArgIndex[level].insert(i);

    // If the original init arg is a memref produced by reinterpret_cast,
    // create a new memref using new strides and offsets created above.
    // This produces a canonicalized memref, which will match what the
    // for loop generates if it modifies the memref. E.g., original
    // reinterpret_cast can produce a memref with const stride:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1] -> (d0 * 256 +
    //  s0 + d1
    //  * s1)>>
    // The new reinterpret_cast will always have dynamic stride and
    // offset:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1
    //  + s0 + d1 * s2)>>
    //
    // For init args that are the first chunk of a modulo pair, there is
    // no need for the type to be rewritten because the strides and
    // offsets are already dynamic.
    if (!moduloInitArgIndices.contains(i) &&
        newInitArgs[i].getDefiningOp<memref::ReinterpretCastOp>()) {
      SmallVector<int64_t> resultShape;
      for (auto s : state.sizes) {
        auto sIntAttr = getIntAttr(s);
        assert(sIntAttr && "expected constant size");
        resultShape.push_back(sIntAttr.value());
      }
      auto castOp = state.createCastOp(resultShape, op.getLoc(), rewriter);

      LLVM_DEBUG({
        llvm::dbgs() << "new reinterpret_cast with dynamic sizes "
                        "and offsets:";
        castOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
        llvm::dbgs() << "\n";
      });

      newInitArgs[i] = castOp.getResult();
    }
  }

  // Pass in the second chunk of each modulo pair
  for (auto &[unrealizedCastOp, chunkData, state] : moduloStates) {
    chunkData[1].initArgIndex = newInitArgs.size();
    newInitArgs.push_back(chunkData[1].reinterpretCast);
  }

  rewriter.restoreInsertionPoint(origIp);

  // Create a new scf::ForOp that uses updated init args and same loop body
  auto newOp = rewriter.create<scf::ForOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
      newInitArgs, [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        IRMapping mapping;
        mapping.map(op.getInductionVar(), iv);
        mapping.map(op.getInitArgs(), newInitArgs);
        mapping.map(op.getRegionIterArgs(), args);

        for (auto &bodyOp : op.getRegion().getOps()) {
          b.clone(bodyOp, mapping);
        }

        // Load op is lowered independent of the pointer, if we have a split
        // pointer due to modulo, we need to "logically combine" these two
        // memrefs into a single one using unrealized_cast_op. This way, when
        // lowering the load, the pattern can detect if additional copies are
        // inserted. When we are in a loop, it is more complicated because we
        // have to insert a new unrealized_cast_op that combines the two memrefs
        // in the init arg list. In addition, because init args hold no offset
        // and size information, we have to manually insert two additional
        // reinterpret_cast ops as input to this unrealized_cast_op so that the
        // load have enough information to generate the corresponding copy.
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(b.getBlock());

        Value zero =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

        for (auto &[unrealizedCastOp, chunkData, state] : moduloStates) {
          SmallVector<Value> newReinterpretCasts;
          for (auto &chunk : chunkData) {
            newReinterpretCasts.push_back(args[chunk.initArgIndex]);
          }

          auto combinedCast = b.create<UnrealizedConversionCastOp>(
              loc, unrealizedCastOp.getResult(0).getType(), newReinterpretCasts,
              unrealizedCastOp->getAttrs());

          args[chunkData[0].initArgIndex].replaceUsesWithIf(
              combinedCast.getResult(0), [](OpOperand &operand) {
                assert(!isa<triton::StoreOp>(operand.getOwner()) &&
                       "Storing to split pointers not supported");
                return isa<triton::LoadOp>(operand.getOwner());
              });
        }
      });

  // Convert the book-keeping data structure to use the correct key and value.
  // Key is converted from init arg index to newly created block arg, and
  // Value's PtrState fields are converted from init arg to newly created block
  // arg
  int cnt = op.getRegionIterArgs().size();
  for (auto [i, state] : knownPtrsTmp) {
    for (auto it = state.offsets.begin(); it != state.offsets.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    for (auto it = state.strides.begin(); it != state.strides.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    auto key = newOp.getRegionIterArgs()[i];
    knownPtrs.insert(std::make_pair(key, state));
  }
  assert(static_cast<size_t>(cnt + moduloStates.size()) ==
             newOp.getRegionIterArgs().size() &&
         "expect to remap all new block args");

  // Replace only the results that correspond to the original scf.for
  auto resultsToReplaceWith = ResultRange(
      newOp.result_begin(), newOp.result_begin() + op.getNumResults());
  rewriter.replaceOp(op, resultsToReplaceWith);

  // Update the loop body. Manually invoke the rewrite logic on addptr and yield
  // in the loop body, so we can take advantage of the states we built up
  for (auto &bodyOp : newOp.getRegion().getOps()) {
    if (auto addptrOp = dyn_cast<triton::AddPtrOp>(bodyOp)) {
      rewriteAddptrOp(addptrOp, rewriter, knownPtrs);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(bodyOp)) {
      rewriteAdvanceOp(advanceOp, rewriter, knownPtrs);
    } else if (auto forOp = dyn_cast<scf::ForOp>(bodyOp)) {
      // TODO:
      //  Nested for loops are not supported at the moment
      assert(0 && "nested loops currently not supported");
      // rewriteForOp(forOp, rewriter, levelToBlockArgIndex, level+1,
      // knownPtrs); levelToBlockArgIndex.erase(level+1);
    }
  }

  if (op.getNumRegionIterArgs()) {
    auto yieldOp = cast<scf::YieldOp>(newOp.getBody()->getTerminator());
    rewriteYieldOp(yieldOp, rewriter, levelToBlockArgIndex, level, knownPtrs);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "new for\n";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

Value PtrAnalysis::getScalarMemRef(Value ptr, Value memRef, const Location loc,
                                   ConversionPatternRewriter &rewriter) {
  assert(cast<triton::PointerType>(ptr.getType()) && "expected scalar pointer");

  // If the pointer is generated by tt.addptr, we will have already inserted an
  // ReinterpretCastOp to cast its type from tt.ptr to unranked memref. Return
  // the result.
  if (ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (auto castOp = memRef.getDefiningOp<memref::ReinterpretCastOp>()) {
      return castOp.getResult();
    } else {
      llvm_unreachable("pointer value is defined by an unexpected op");
    }
  }

  assert(isa<BlockArgument>(ptr) &&
         "pointer is neither produced by addptr nor a block argument");
  PtrState state;
  state.source = memRef;
  state.offsets.push_back(rewriter.getIndexAttr(0));
  state.sizes.push_back(rewriter.getIndexAttr(1));
  state.strides.push_back(rewriter.getIndexAttr(1));
  state.modulos.push_back(std::nullopt);
  auto castOp = state.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
  return castOp.getResult();
}

} // namespace triton
} // namespace mlir
