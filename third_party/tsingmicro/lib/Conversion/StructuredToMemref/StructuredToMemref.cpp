//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "Address/Dialect/IR/AddressDialect.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), source,
                                     offsets, dims, strides);
}

static OpFoldResult accumulateTargetOffset(tts::MakeTensorPtrOp op,
                                           OpBuilder &b) {
  Location loc = op->getLoc();
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (auto o : op.getMixedOffsets()) {
    targetOffset = addOFRs(targetOffset, o, loc, b);
  }
  return targetOffset;
}

namespace {

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  // If there are dimensions with size 1 and stride 0, replace 0 stride with
  // the product of sizes of all lower dimensions. This avoids creating memref
  // with zero stride.
  static llvm::SmallVector<OpFoldResult>
  getMixedStridesForMemref(tts::MakeTensorPtrOp op, OpBuilder &b) {
    llvm::SmallVector<OpFoldResult> strides;
    auto accumulate = 1;
    for (auto [size, stride] :
         llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
      auto strideIntAttr = getIntAttr(stride);
      if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
        strides.push_back(b.getIndexAttr(accumulate));
      } else if (auto v = llvm::dyn_cast_if_present<Value>(stride)) {
        OpFoldResult result = getAsOpFoldResult(v);
        strides.push_back(result);
      } else {
        strides.push_back(stride);
      }
      accumulate *= size;
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset = accumulateTargetOffset(op, rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(
        op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
        resultShape);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), resultType, adaptor.getBase(), targetOffset,
        op.getMixedSizes(), mixedStrides);

    rewriter.replaceOp(op, castOp);

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  MakeTensorPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Order is a compiler hint. We can optimize data load/store according
    // the order attribute.
    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) {
      return success();
    }

    return failure();
  }
};

memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                ArrayRef<OpFoldResult> sizes,
                                ArrayRef<OpFoldResult> strides, Location loc,
                                ConversionPatternRewriter &rewriter) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), src,
                                            offsets, sizes, strides);
}

Value createCastOps(tts::MakeTensorPtrOp op,
                    ConversionPatternRewriter &rewriter, Value start,
                    SmallVector<Value> sizesValues,
                    SmallVector<Value> strideVals) {

  Type elemType =
      cast<triton::PointerType>(op.getBase().getType()).getPointeeType();

  auto unrankedMemrefType = UnrankedMemRefType::get(elemType, 0);
  // WARNING: TypeConverter cannot automatically insert
  // `UnrealizedConversionCastOp` through the materialization mechanism.
  auto unrankedMemref = rewriter
                            .create<UnrealizedConversionCastOp>(
                                op->getLoc(), unrankedMemrefType, op.getBase())
                            ->getResults()[0];

  auto layout = StridedLayoutAttr::get(
      op.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(sizesValues.size(), ShapedType::kDynamic));
  MemRefType resultType = MemRefType::get(
      SmallVector<int64_t>(sizesValues.size(), ShapedType::kDynamic), elemType,
      layout);
  auto block = rewriter.create<memref::ReinterpretCastOp>(
      op->getLoc(), resultType, unrankedMemref, start, sizesValues, strideVals);
  return block;
}

std::pair<memref::SubViewOp, Value>
getMemSubviews(SmallVector<OpFoldResult> &dims, Value block, Location loc,
               int64_t splitDim, ConversionPatternRewriter &rewriter) {

  auto rank = dims.size();
  OpFoldResult maskSize =
      rewriter.create<memref::DimOp>(loc, block, splitDim).getResult();

  OpFoldResult subviewDimFull = dims[splitDim];
  OpFoldResult subviewDim = minOFRs(maskSize, subviewDimFull, loc, rewriter);

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  SmallVector<OpFoldResult> sizes(dims.begin(), dims.end());
  sizes[splitDim] = subviewDim;

  auto sv = createSubview(block, offsets, sizes, strides, loc, rewriter);
  auto remainMask = rewriter.create<arith::SubIOp>(
      loc, ofrToIndexValue(subviewDimFull, loc, rewriter),
      ofrToIndexValue(subviewDim, loc, rewriter));

  return {sv, remainMask};
}

void createMemCopies(Value block, Value dst, Location loc,
                     ConversionPatternRewriter &rewriter, Value &dstOffset,
                     int64_t splitDim, bool isLoadToDst) {
  auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

  auto one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

  auto rank = cast<MemRefType>(dst.getType()).getRank();
  SmallVector<Value> blockShape;
  for (int i = 0; i < rank; i++) {
    blockShape.push_back(rewriter.create<memref::DimOp>(loc, block, i));
  }

  SmallVector<Value> dstOffsets(rank, zero);
  dstOffsets[splitDim] = dstOffset;

  auto blockDst =
      rewriter.create<memref::SubViewOp>(loc, dst,
                                         /* offsets */
                                         dstOffsets,
                                         /* sizes */
                                         blockShape,
                                         /* strides */
                                         SmallVector<Value>(rank, one));
  dstOffset =
      rewriter.create<arith::AddIOp>(loc, dstOffset, blockShape[splitDim]);

  if (isLoadToDst) {
    rewriter.create<memref::CopyOp>(loc, block, blockDst);
  } else {
    rewriter.create<memref::CopyOp>(loc, blockDst, block);
  }
}

Value processMemSubviewCopies(Location loc, ConversionPatternRewriter &rewriter,
                              Value alloc, Value block,
                              SmallVector<OpFoldResult> mixedDims,
                              Value &allocOffset, int64_t splitDim,
                              bool isLoadToDst) {

  Value subview;
  Value remainMask;
  if (mixedDims.empty()) {
    subview = block;
  } else {
    auto res = getMemSubviews(mixedDims, block, loc, splitDim, rewriter);
    subview = res.first;
    remainMask = res.second;
  }

  createMemCopies(subview, alloc, loc, rewriter, allocOffset, splitDim,
                  isLoadToDst);
  return remainMask;
}

void rewriteSideBySideMemAccess(tts::MakeTensorPtrOp makeTensorPtrOp,
                                ConversionPatternRewriter &rewriter,
                                Value alloc,
                                SmallVector<OpFoldResult> mixedDims,
                                bool isLoadToDst) {
  assert(makeTensorPtrOp.getStaticShape().size() == 1 ||
         makeTensorPtrOp.getStaticShape()[0] == 0);
  auto loc = makeTensorPtrOp->getLoc();
  auto targetOffset = ofrToIndexValue(
      accumulateTargetOffset(makeTensorPtrOp, rewriter), loc, rewriter);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Handling side-by-side wraparound
  //
  // Same limitations apply to the stacked wraparound case.
  //
  ////////////////////////////////////////////////////////////////////////////
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
  //    offset_dim_0  = scaled_offset_0
  //    offset_dim_1 = scaled_offset_1
  //    col_start = scaled_offset_1 % N
  //    remainSize = colSize
  //    size =  N - col_start
  //    while (remainSize > size):
  //       reinterpret (col_start,  size,  stride )
  //       remainSize = remainSize - size
  //       col_start = (scaled_offset_0 + size) %N
  //       size =  N
  //
  //    reinterpret (col_start,  remainSize,  stride )
  //
  ////////////////////////////////////////////////////////////////////////////
  auto rank = cast<RankedTensorType>(makeTensorPtrOp.getType()).getRank();
  SmallVector<Value> scaledOffset(rank);
  auto offsets = makeTensorPtrOp.getMixedOffsets();
  std::transform(offsets.begin(), offsets.end(), scaledOffset.begin(),
                 [&](auto val) { return ofrToIndexValue(val, loc, rewriter); });
  auto lastDim = rank - 1;

  assert(rank == makeTensorPtrOp.getSizes().size());
  // Data block shape to be read
  auto sizesInt = makeTensorPtrOp.getSizes();
  SmallVector<Value> sizesValues(rank);
  std::transform(sizesInt.begin(), sizesInt.end(), sizesValues.begin(),
                 [&](auto val) {
                   return rewriter.create<arith::ConstantOp>(
                       loc, rewriter.getIndexAttr(val));
                 });
  // Total side by side size
  Value totalSize = sizesValues.back();

  // NOTE: We use `scaledOffset[lastdim]` for modulo because the loop and the
  // modulo dimension cannot be in the same dimension (otherwise ptrAnalysis
  // cannot analyze `make_tptr` of `splitMemory`).
  Value N =
      ofrToIndexValue(makeTensorPtrOp.getMixedShape()[lastDim], loc, rewriter);
  Value x = rewriter.create<arith::RemSIOp>(loc, scaledOffset[lastDim], N);
  Value y =
      rewriter.create<arith::SubIOp>(loc, targetOffset, scaledOffset[lastDim]);
  SmallVector<Value> strideVals =
      ofrsToIndexValues(makeTensorPtrOp.getMixedStrides(), loc, rewriter);

  Value remainSize = totalSize;
  Value size = rewriter.create<arith::SubIOp>(loc, N, x);
  Value colStart = rewriter.create<arith::AddIOp>(loc, y, x);
  Value allocOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Type> typeR{remainSize.getType(), size.getType(),
                          colStart.getType(), allocOffset.getType()};
  SmallVector<Value> valueR{remainSize, size, colStart, allocOffset};
  if (!mixedDims.empty()) {
    Value mixedDim = ofrsToIndexValues(mixedDims[lastDim], loc, rewriter)[0];
    typeR.push_back(mixedDim.getType());
    valueR.push_back(mixedDim);
  }
  auto whileOp = rewriter.create<scf::WhileOp>(
      loc, typeR, valueR,
      /*beforeBuilder=*/
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             args[0], args[1]);
        b.create<scf::ConditionOp>(loc, cond, args);
      },
      /*afterBuilder=*/
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value remainSize = args[0];
        Value size = args[1];
        Value colStart = args[2];
        Value allocOffset = args[3];
        sizesValues[lastDim] = size;
        if (!mixedDims.empty()) {
          Value mixedDim = args[4];
          mixedDims[lastDim] = mixedDim;
        }
        Value block = createCastOps(makeTensorPtrOp, rewriter, colStart,
                                    sizesValues, strideVals);
        Value mixedDim =
            processMemSubviewCopies(loc, rewriter, alloc, block, mixedDims,
                                    allocOffset, lastDim, isLoadToDst);
        remainSize = b.create<arith::SubIOp>(loc, remainSize, size);
        colStart = b.create<arith::AddIOp>(loc, colStart, size);
        colStart = b.create<arith::RemSIOp>(loc, colStart, N);
        colStart = b.create<arith::AddIOp>(loc, colStart, y);
        size = N;
        SmallVector<Value> newArgs{remainSize, size, colStart, allocOffset};
        if (!mixedDims.empty()) {
          newArgs.push_back(mixedDim);
        }
        b.create<scf::YieldOp>(loc, newArgs);
      });

  remainSize = whileOp->getResult(0);
  colStart = whileOp->getResult(2);
  sizesValues[lastDim] = remainSize;
  allocOffset = whileOp->getResult(3);
  if (!mixedDims.empty()) {
    mixedDims[lastDim] = whileOp->getResult(4);
  }

  Value block = createCastOps(makeTensorPtrOp, rewriter, colStart, sizesValues,
                              strideVals);
  processMemSubviewCopies(loc, rewriter, alloc, block, mixedDims, allocOffset,
                          lastDim, isLoadToDst);
}

void rewriteStackedMemAccess(tts::MakeTensorPtrOp makeTensorPtrOp,
                             ConversionPatternRewriter &rewriter, Value alloc,
                             SmallVector<OpFoldResult> mixedDims,
                             bool isLoadToDst) {
  assert(makeTensorPtrOp.getStaticShape()[1] == 0);

  auto loc = makeTensorPtrOp->getLoc();
  auto resultShape =
      cast<RankedTensorType>(makeTensorPtrOp.getType()).getShape();

  assert(resultShape.size() == 2);
  auto rank = cast<RankedTensorType>(makeTensorPtrOp.getType()).getRank();
  auto targetOffset = ofrToIndexValue(
      accumulateTargetOffset(makeTensorPtrOp, rewriter), loc, rewriter);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Handling stacked wraparound
  // See side-by-side wraparound for details.
  //
  ////////////////////////////////////////////////////////////////////////////
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
  //                  ~~~~~~~~~~~~~~~~~
  //                         ^
  //                         |
  //          We have already computed
  //          rows * strideRows = modRow = shape[1]
  //          in TritonToStructured
  //
  //          clampedOff - targetOffset
  //    d1 = --------------------
  //              strideRows
  //
  //    N = stride[0]
  //    M = shape[0] % N
  //    row_start = targetOffset / N % M
  //    remainSize = rowsize
  //    size =  M - row_start
  //    start = row_start * N + targetOffset % N
  //    while (remainSize > size):
  //       reinterpret (start,  size,  stride )
  //       remainSize = remainSize - size
  //       start = scaled_offset_1
  //       size =  M
  //    reinterpret (start,  remainSize,  stride )
  ////////////////////////////////////////////////////////////////////////////
  //                       cols
  //
  //               wrappedAroundOff
  //      --------------*------------*--------
  //      |                                  |
  //      |           targetOffset           |
  //      |             *------------|       |
  //      |             |            |       |
  //      |             |            |       |
  //  rows|    rowSize  |            |       |
  //      |             |            |       |
  //      |             |            |       |
  //      |             *------------|       |
  //      |          nextOff                 |
  //      |                                  |
  //      |          clampedOff              |
  //      --------------*---------------------
  //
  //    d1 = rowSize
  //
  //    d2 = 0
  Value modM = makeTensorPtrOp.getShape()[0];
  Value N =
      ofrToIndexValue(makeTensorPtrOp.getMixedStrides()[0], loc, rewriter);

  auto sizesInt = makeTensorPtrOp.getSizes();
  SmallVector<Value> sizesValues(rank);
  std::transform(sizesInt.begin(), sizesInt.end(), sizesValues.begin(),
                 [&](auto val) {
                   return rewriter.create<arith::ConstantOp>(
                       loc, rewriter.getIndexAttr(val));
                 });
  SmallVector<Value> strideVals =
      ofrsToIndexValues(makeTensorPtrOp.getMixedStrides(), loc, rewriter);

  // NOTE: Here, we need to use `targetOffset` for integer division, instead of
  // `scaledOffset[1]` as in `sidebyside`. This is because the offset of the
  // column loop analyzed by ptrAnalysis is added to `scaledOffset[0]` (row),
  // and we assume that the offset in the 1-dimensional dimension must be less
  // than N.
  Value M = rewriter.create<arith::DivSIOp>(loc, modM, N);
  Value rowStart = rewriter.create<arith::DivSIOp>(loc, targetOffset, N);
  rowStart = rewriter.create<arith::RemSIOp>(loc, rowStart, M);
  Value colStart = rewriter.create<arith::RemSIOp>(loc, targetOffset, N);

  Value remainSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexAttr(makeTensorPtrOp.getSizes()[0]));
  Value size = rewriter.create<arith::SubIOp>(loc, M, rowStart);
  Value start = rewriter.create<arith::MulIOp>(loc, rowStart, N);
  start = rewriter.create<arith::AddIOp>(loc, start, colStart);
  Value allocOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Type> typeR{remainSize.getType(), size.getType(), start.getType(),
                          allocOffset.getType()};
  SmallVector<Value> valueR{remainSize, size, start, allocOffset};
  if (!mixedDims.empty()) {
    Value mixedDim = ofrsToIndexValues(mixedDims[0], loc, rewriter)[0];
    typeR.push_back(mixedDim.getType());
    valueR.push_back(mixedDim);
  }
  auto whileOp = rewriter.create<scf::WhileOp>(
      loc, typeR, valueR,
      /*beforeBuilder=*/
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             args[0], args[1]);
        b.create<scf::ConditionOp>(loc, cond, args);
      },
      /*afterBuilder=*/
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value remainSize = args[0];
        Value size = args[1];
        Value start = args[2];
        Value allocOffset = args[3];
        sizesValues[0] = size;
        if (!mixedDims.empty()) {
          Value mixedDim = args[4];
          mixedDims[0] = mixedDim;
        }
        Value block = createCastOps(makeTensorPtrOp, rewriter, start,
                                    sizesValues, strideVals);
        Value mixedDim = processMemSubviewCopies(
            loc, rewriter, alloc, block, mixedDims, allocOffset,
            0 /*dim of mod row*/, isLoadToDst);
        remainSize = b.create<arith::SubIOp>(loc, remainSize, size);
        Value addOffsets = b.create<arith::MulIOp>(loc, size, N);
        start = b.create<arith::AddIOp>(loc, start, addOffsets);
        start = b.create<arith::RemSIOp>(loc, start, modM);
        size = M;
        SmallVector<Value> newArgs{remainSize, size, start, allocOffset};
        if (!mixedDims.empty()) {
          newArgs.push_back(mixedDim);
        }
        b.create<scf::YieldOp>(loc, newArgs);
      });
  remainSize = whileOp->getResult(0);
  start = whileOp->getResult(2);
  sizesValues[0] = remainSize;
  allocOffset = whileOp->getResult(3);
  if (!mixedDims.empty()) {
    mixedDims[0] = whileOp->getResult(4);
  }
  Value block =
      createCastOps(makeTensorPtrOp, rewriter, start, sizesValues, strideVals);
  processMemSubviewCopies(loc, rewriter, alloc, block, mixedDims, allocOffset,
                          0 /*dim of mod row*/, isLoadToDst);
}

void rewriteMakeTensorPtrAndMemAccess(tts::MakeTensorPtrOp makeTensorPtrOp,
                                      ConversionPatternRewriter &rewriter,
                                      Value alloc,
                                      SmallVector<OpFoldResult> mixedDims,
                                      bool isLoadToDst) {
  auto parentShape = makeTensorPtrOp.getStaticShape();
  if (parentShape.size() > 1 && parentShape[0] == ShapedType::kDynamic) {
    rewriteStackedMemAccess(makeTensorPtrOp, rewriter, alloc, mixedDims,
                            isLoadToDst);
  } else {
    rewriteSideBySideMemAccess(makeTensorPtrOp, rewriter, alloc, mixedDims,
                               isLoadToDst);
  }
}

tts::MakeTensorPtrOp isSplitMemoryAccess(Operation *op) {
  auto makeTensorPtrOp =
      op->getOperand(0).getDefiningOp<tts::MakeTensorPtrOp>();
  if (makeTensorPtrOp && makeTensorPtrOp.isSplitPtr()) {
    return makeTensorPtrOp;
  }
  return nullptr;
}

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto other = op.getOther();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    Value alloc;
    MemRefType memrefType = MemRefType::get(tensorType.getShape(), elemType);

    // No mask
    assert(!other && "other value used in non-masked load");

    if (auto makeTensorPtrOp = isSplitMemoryAccess(op)) {
      alloc = rewriter.create<memref::AllocOp>(loc, memrefType);
      rewriteMakeTensorPtrAndMemAccess(makeTensorPtrOp, rewriter, alloc,
                                       SmallVector<OpFoldResult>{}, true);
    } else {
      alloc = rewriter.create<bufferization::CloneOp>(loc, memrefType, ptr);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();

    // Fill load destination with other value
    auto other = op.getOther();
    if (!other) {

      LLVM_DEBUG(op->emitRemark(
          "Masked load without other value, using zero padding instead\n"));
      // FIXME: Different reduction op need different reduce base value
      other = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getZeroAttr(elemType));
    }

    // For each dimension check if dims[i] < shape[i], or-accumulate
    // the result
    auto shape = tensorType.getShape();
    auto accBase =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
            .getResult();
    for (size_t i = 0; i < shape.size(); i++) {
      auto shapei = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(shape[i]));

      Value dimi = dyn_cast<Value>(mixedDims[i]);
      if (!dimi) {
        dimi = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(op.getStaticMaskDims()[i]));
      }

      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 dimi, shapei);
      accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmp);
    }

    // condition the memset on the or-accumulation
    // initialize with padding prior to CopyOp
    rewriter.create<scf::IfOp>(loc, accBase, [&](OpBuilder &b, Location loc) {
      b.create<linalg::FillOp>(loc, ValueRange{other}, ValueRange{alloc});
      b.create<scf::YieldOp>(loc);
    });

    if (auto makeTensorPtrOp = isSplitMemoryAccess(op)) {
      rewriteMakeTensorPtrAndMemAccess(makeTensorPtrOp, rewriter, alloc,
                                       mixedDims, true);
    } else {
      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), mixedDims, ptr, loc, rewriter);
      memref::SubViewOp dstSubview =
          getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

public:
  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.hasMask()) {
      return rewriteMaskedLoad(op, adaptor, rewriter);
    } else {
      return rewriteStructuredLoad(op, adaptor, rewriter);
    }
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

  LogicalResult rewriteMaskedStore(tts::StoreOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto storeValue = op.getValue();
    auto tensorType = cast<RankedTensorType>(storeValue.getType());
    auto elemType = tensorType.getElementType();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();

    auto mixedDims = op.getMixedMaskDims();
    if (auto makeTensorPtrOp = isSplitMemoryAccess(op)) {
      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto srcType = cast<RankedTensorType>(srcSlice.getType());
      auto srcSliceMemRef = rewriter.create<bufferization::ToMemrefOp>(
          loc, MemRefType::get(srcType.getShape(), srcType.getElementType()),
          srcSlice);
      rewriteMakeTensorPtrAndMemAccess(makeTensorPtrOp, rewriter,
                                       srcSliceMemRef, mixedDims, false);
    } else {
      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  rewriteStructuredStore(tts::StoreOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto storeValue = op.getValue();

    auto tensorType = cast<RankedTensorType>(storeValue.getType());
    auto elemType = tensorType.getElementType();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();
    MemRefType memrefType = MemRefType::get(tensorType.getShape(), elemType);

    if (auto makeTensorPtrOp = isSplitMemoryAccess(op)) {
      auto srcMemRef = rewriter.create<bufferization::ToMemrefOp>(
          loc, memrefType, storeValue);
      rewriteMakeTensorPtrAndMemAccess(makeTensorPtrOp, rewriter, srcMemRef,
                                       SmallVector<OpFoldResult>{}, false);
    } else {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.hasMask()) {
      return rewriteMaskedStore(op, adaptor, rewriter);
    } else {
      return rewriteStructuredStore(op, adaptor, rewriter);
    }
  }
};

struct AtomicRMWOpConverter : public OpConversionPattern<tts::AtomicRMWOp> {
private:
  using OpConversionPattern<tts::AtomicRMWOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

public:
  LogicalResult
  matchAndRewrite(tts::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto value = adaptor.getValue();

    auto type = cast<RankedTensorType>(value.getType());
    auto rank = type.getRank();

    Value init = rewriter.create<tensor::EmptyOp>(loc, type.getShape(),
                                                  type.getElementType());

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto valueSlice = getExtractSlice(rank, mixedDims, value, loc, rewriter);
      auto ptrSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto atomicRMWOp = rewriter.create<mk::AtomicRMWOp>(
          loc, op.getType(), ptrSubview, valueSlice, init,
          op.getAtomicRmwOpAttr(), op.getSemAttr(), op.getScopeAttr());
      rewriter.replaceOp(op, atomicRMWOp);
    } else {
      auto atomicRMWOp = rewriter.create<mk::AtomicRMWOp>(
          loc, op.getType(), ptr, value, init, op.getAtomicRmwOpAttr(),
          op.getSemAttr(), op.getScopeAttr());
      rewriter.replaceOp(op, atomicRMWOp);
    }
    return success();
  }
};

struct AtomicCASOpConverter : public OpConversionPattern<tts::AtomicCASOp> {
private:
  using OpConversionPattern<tts::AtomicCASOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(tts::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getOffset())
      return failure();

    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto cmp = adaptor.getCmp();
    auto value = adaptor.getValue();

    auto type = cast<RankedTensorType>(value.getType());

    Value init = rewriter.create<tensor::EmptyOp>(loc, type.getShape(),
                                                  type.getElementType());

    auto atomicCASOp = rewriter.create<mk::AtomicCASOp>(
        loc, op.getType(), ptr, cmp, value, init, op.getSemAttr(),
        op.getScopeAttr());
    rewriter.replaceOp(op, atomicCASOp);

    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MakeTensorPtrConverter>(typeConverter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter, AtomicRMWOpConverter,
               AtomicCASOpConverter>(patterns.getContext());
}
