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

#include "TritonToLinalg/DescriptorConverter.h"
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToLinalg/TritonOpConverter.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace DescriptorConverter {
using namespace mlir;
using namespace triton;

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

Descriptor unpackDescriptor(TensorDescType type, Value desc,
                            ConversionPatternRewriter &rewriter) {
  auto makeDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
  assert(makeDescOp && "Descriptor must be defined by MakeTensorDescOp");

  Descriptor res;

  res.base = makeDescOp.getBase();
  for (auto s : makeDescOp.getShape()) {
    res.shape.push_back(rewriter.createOrFold<arith::ExtSIOp>(
        makeDescOp.getLoc(), rewriter.getI64Type(), s));
  }
  for (auto st : makeDescOp.getStrides()) {
    res.strides.push_back(rewriter.createOrFold<arith::ExtSIOp>(
        makeDescOp.getLoc(), rewriter.getI64Type(), st));
  }
  res.padding = makeDescOp.getPaddingAttr();

  return res;
}

SmallVector<int32_t> computeOrder(ArrayRef<int64_t> shape) {
  SmallVector<int32_t> order;
  int rank = shape.size();
  order.reserve(rank);
  // default by [dims - 1, ..., 0]
  for (int i = rank - 1; i >= 0; --i) {
    order.push_back(i);
  }
  return order;
}

DenseI32ArrayAttr getFullBoundaryCheckAttr(ConversionPatternRewriter &rewriter,
                                           ArrayRef<int64_t> shape) {
  SmallVector<int32_t> boundaryCheck;
  boundaryCheck.reserve(shape.size());
  for (int32_t dim = 0; dim < static_cast<int32_t>(shape.size()); ++dim) {
    boundaryCheck.push_back(dim);
  }
  return rewriter.getDenseI32ArrayAttr(boundaryCheck);
}

Value expandOffsets(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockShape, Value offsets, unsigned dim) {
  Value expandedResult = offsets;
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == dim) {
      continue;
    }
    expandedResult =
        builder.create<triton::ExpandDimsOp>(loc, expandedResult, j);
  }

  return expandedResult;
}

Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 Value offset, unsigned dim) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI64Type());
  Value splatOffset =
      builder.create<triton::SplatOp>(loc, indexRowType, offset);
  Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                    blockShape[dim]);
  Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

  Value offsets = builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, desc.base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = builder.create<triton::SplatOp>(
        loc, offsets[i].getType(), desc.strides[i]);
    Value offsetWithStride =
        builder.create<arith::MulIOp>(loc, offsets[i], splatStride);
    Value broadcasted = builder.create<triton::BroadcastOp>(
        loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
}

Value generateMaskFromOffsetRanges(OpBuilder &builder, const Location &loc,
                                   ArrayRef<std::int64_t> blockShape,
                                   Descriptor &desc, ValueRange offsetRanges) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsetRanges.size());

  // Generate mask per dimension
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  Value mask;
  for (std::size_t i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange = offsetRanges[i];

    // Compare with lower bound
    Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
        loc, builder.getI64Type(), 0);
    Value splatLowerBound = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), lowerBound);
    Value cmpLower = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

    // Compare with upper bound
    Value splatUpperBound = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), desc.shape[i]);
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

Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                   ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                      offsetRanges);
}

SmallVector<mlir::Value> castToI64(OpBuilder &builder,
                                   mlir::ValueRange values) {
  auto i64Type = builder.getI64Type();
  return llvm::map_to_vector(values, [&](mlir::Value v) {
    return builder.createOrFold<arith::ExtSIOp>(v.getLoc(), i64Type, v);
  });
}

LogicalResult DescriptorLoadConverter::matchAndRewrite(
    triton::DescriptorLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  const auto blockShape = op.getDesc().getType().getBlockType().getShape();
  auto descTy = op.getDesc().getType();
  auto indices = op.getIndices();

  // 1. unpack descriptor
  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

  // 2. create make_tensor_ptr
  SmallVector<int32_t> tensorShapeValues;
  for (auto dim : blockShape) {
    tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }
  Value tensorPtr =
      rewriter.create<triton::MakeTensorPtrOp>(loc,
                                               desc.base,         // base
                                               desc.shape,        // shape
                                               desc.strides,      // strides
                                               indices,           // offset
                                               tensorShapeValues, // tensorShape
                                               computeOrder(blockShape) // order
      );
  // 3. replace tt.load
  auto boundaryCheck = getFullBoundaryCheckAttr(rewriter, blockShape);
  triton::PaddingOptionAttr padding = desc.padding;
  auto cache = triton::CacheModifierAttr::get(rewriter.getContext(),
                                              triton::CacheModifier::NONE);
  auto evict = triton::EvictionPolicyAttr::get(rewriter.getContext(),
                                               triton::EvictionPolicy::NORMAL);
  auto isVolatile = rewriter.getBoolAttr(false);

  if (auto a = op->getAttrOfType<triton::CacheModifierAttr>("cache"))
    cache = a;
  if (auto a = op->getAttrOfType<triton::EvictionPolicyAttr>("evict"))
    evict = a;
  if (auto a = op->getAttrOfType<BoolAttr>("isVolatile"))
    isVolatile = a;

  auto newLoad = rewriter.create<triton::LoadOp>(
      loc, descTy.getSignlessBlockType(), tensorPtr,
      Value(), // mask
      Value(), // other
      boundaryCheck, padding, cache, evict, isVolatile);

  rewriter.replaceOp(op, newLoad.getResult());

  return success();
}

LogicalResult DescriptorStoreConverter::matchAndRewrite(
    triton::DescriptorStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  const auto blockShape = op.getDesc().getType().getBlockType().getShape();
  auto descTy = op.getDesc().getType();
  auto indices = op.getIndices();

  // 1. unpack descriptor
  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

  // 2. create make_tensor_ptr
  SmallVector<int32_t> tensorShapeValues;
  for (auto dim : blockShape) {
    tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }
  Value tensorPtr =
      rewriter.create<triton::MakeTensorPtrOp>(loc,
                                               desc.base,         // base
                                               desc.shape,        // shape
                                               desc.strides,      // strides
                                               indices,           // offset
                                               tensorShapeValues, // tensorShape
                                               computeOrder(blockShape) // order
      );

  // 3. replace tt.store
  Value valueToStore = adaptor.getSrc();

  auto maskType = RankedTensorType::get(blockShape, rewriter.getI1Type());
  rewriter.create<arith::ConstantOp>(loc,
                                     DenseElementsAttr::get(maskType, true));
  auto boundaryCheck = getFullBoundaryCheckAttr(rewriter, blockShape);
  auto cacheModifier = triton::CacheModifierAttr::get(
      rewriter.getContext(), triton::CacheModifier::NONE);
  auto evictionPolicy = triton::EvictionPolicyAttr::get(
      rewriter.getContext(), triton::EvictionPolicy::NORMAL);

  auto newStore = rewriter.create<triton::StoreOp>(loc, tensorPtr, valueToStore,
                                                   Value(), // mask
                                                   boundaryCheck, cacheModifier,
                                                   evictionPolicy);

  rewriter.eraseOp(op);
  return success();
}

LogicalResult DescriptorScatterConverter::matchAndRewrite(
    triton::DescriptorScatterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto descTy = op.getDesc().getType();
  auto srcType = cast<RankedTensorType>(op.getSrc().getType());
  const auto rowBlockShape = descTy.getSignlessBlockType().getShape();

  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);
  SmallVector<int32_t> tensorShapeValues;
  tensorShapeValues.reserve(rowBlockShape.size());
  for (auto dim : rowBlockShape) {
    tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }

  auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowUpperBound;
  if (srcType.isDynamicDim(0)) {
    rowUpperBound = rewriter.create<tensor::DimOp>(loc, adaptor.getSrc(), 0);
  } else {
    rowUpperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, srcType.getShape()[0]);
  }
  auto rowBoundaryCheck = getFullBoundaryCheckAttr(rewriter, rowBlockShape);
  auto cacheModifier = triton::CacheModifierAttr::get(
      rewriter.getContext(), triton::CacheModifier::NONE);
  auto evictionPolicy = triton::EvictionPolicyAttr::get(
      rewriter.getContext(), triton::EvictionPolicy::NORMAL);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zeroIndex, rowUpperBound, oneIndex, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value rowIv,
          ValueRange) {
        Value xOffset = nestedBuilder.create<tensor::ExtractOp>(
            nestedLoc, adaptor.getXOffsets(), ValueRange{rowIv});
        Value tensorPtr = nestedBuilder.create<triton::MakeTensorPtrOp>(
            nestedLoc, desc.base, desc.shape, desc.strides,
            ValueRange{xOffset, adaptor.getYOffset()}, tensorShapeValues,
            computeOrder(rowBlockShape));
        SmallVector<OpFoldResult> extractOffsets{rowIv,
                                                 nestedBuilder.getIndexAttr(0)};
        SmallVector<OpFoldResult> extractSizes{
            nestedBuilder.getIndexAttr(rowBlockShape[0]),
            nestedBuilder.getIndexAttr(rowBlockShape[1])};
        SmallVector<OpFoldResult> extractStrides{nestedBuilder.getIndexAttr(1),
                                                 nestedBuilder.getIndexAttr(1)};
        auto rowValue = nestedBuilder.create<tensor::ExtractSliceOp>(
            nestedLoc, adaptor.getSrc(), extractOffsets, extractSizes,
            extractStrides);
        auto rowStore = nestedBuilder.create<triton::StoreOp>(
            nestedLoc, tensorPtr, rowValue.getResult(), Value(),
            rowBoundaryCheck, cacheModifier, evictionPolicy);
        rowStore->setAttr(ConverterUtils::discreteAttrName,
                          nestedBuilder.getUnitAttr());
        nestedBuilder.create<scf::YieldOp>(nestedLoc);
      });
  loop->setAttr("ExtractedLoadOrStore", rewriter.getUnitAttr());
  loop->setAttr("hivm.parallel_loop", rewriter.getUnitAttr());

  rewriter.eraseOp(op);
  return success();
}

SmallVector<NamedAttribute> filterSegmentSizes(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> filteredAttrs;
  llvm::copy_if(attrs, std::back_inserter(filteredAttrs),
                [](const NamedAttribute &attr) {
                  return attr.getName().getValue() != "operandSegmentSizes";
                });
  return filteredAttrs;
}

LogicalResult DescriptorGatherConverter::matchAndRewrite(
    triton::DescriptorGatherOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto descTy = cast<TensorDescType>(op.getDesc().getType());
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  const auto blockShape = resultType.getShape();
  const auto rowBlockShape = descTy.getSignlessBlockType().getShape();

  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);
  SmallVector<int32_t> tensorShapeValues;
  tensorShapeValues.reserve(rowBlockShape.size());
  for (auto dim : rowBlockShape) {
    tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }

  auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowUpperBound =
      rewriter.create<tensor::DimOp>(loc, adaptor.getXOffsets(), zeroIndex);
  auto rowBoundaryCheck = getFullBoundaryCheckAttr(rewriter, rowBlockShape);
  auto cache = triton::CacheModifierAttr::get(rewriter.getContext(),
                                              triton::CacheModifier::NONE);
  auto evict = triton::EvictionPolicyAttr::get(rewriter.getContext(),
                                               triton::EvictionPolicy::NORMAL);
  auto isVolatile = rewriter.getBoolAttr(false);

  if (auto attr = op->getAttrOfType<triton::CacheModifierAttr>("cache"))
    cache = attr;
  if (auto attr = op->getAttrOfType<triton::EvictionPolicyAttr>("evict"))
    evict = attr;
  if (auto attr = op->getAttrOfType<BoolAttr>("isVolatile"))
    isVolatile = attr;

  SmallVector<Value> dynamicResultSizes;
  dynamicResultSizes.reserve(resultType.getNumDynamicDims());
  for (const auto &[dim, size] : llvm::enumerate(blockShape)) {
    if (!ShapedType::isDynamic(size))
      continue;
    if (dim == 0) {
      dynamicResultSizes.push_back(rowUpperBound);
      continue;
    }
    dynamicResultSizes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, rowBlockShape[dim]));
  }

  auto initialTensor = rewriter.create<tensor::EmptyOp>(
      loc, blockShape, resultType.getElementType(), dynamicResultSizes);
  auto loop = rewriter.create<scf::ForOp>(
      loc, zeroIndex, rowUpperBound, oneIndex,
      ValueRange{initialTensor.getResult()},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value rowIv,
          ValueRange iterArgs) {
        Value xOffset = nestedBuilder.create<tensor::ExtractOp>(
            nestedLoc, adaptor.getXOffsets(), ValueRange{rowIv});
        Value tensorPtr = nestedBuilder.create<triton::MakeTensorPtrOp>(
            nestedLoc, desc.base, desc.shape, desc.strides,
            ValueRange{xOffset, adaptor.getYOffset()}, tensorShapeValues,
            computeOrder(rowBlockShape));
        auto rowLoad = nestedBuilder.create<triton::LoadOp>(
            nestedLoc, descTy.getSignlessBlockType(), tensorPtr,
            Value(), // mask
            Value(), // other
            rowBoundaryCheck, desc.padding, cache, evict, isVolatile);
        for (const auto &attr : filterSegmentSizes(op->getAttrs())) {
          if (!rowLoad->hasAttr(attr.getName())) {
            rowLoad->setAttr(attr.getName(), attr.getValue());
          }
        }
        rowLoad->setAttr(ConverterUtils::discreteAttrName,
                         nestedBuilder.getUnitAttr());

        auto insertSlice = nestedBuilder.create<tensor::InsertSliceOp>(
            nestedLoc, rowLoad.getResult(), iterArgs[0],
            SmallVector<OpFoldResult>{rowIv, nestedBuilder.getIndexAttr(0)},
            SmallVector<OpFoldResult>{
                nestedBuilder.getIndexAttr(rowBlockShape[0]),
                nestedBuilder.getIndexAttr(rowBlockShape[1])},
            SmallVector<OpFoldResult>{nestedBuilder.getIndexAttr(1),
                                      nestedBuilder.getIndexAttr(1)});
        insertSlice->setAttr(ConverterUtils::discreteAttrName,
                             nestedBuilder.getUnitAttr());
        nestedBuilder.create<scf::YieldOp>(nestedLoc, insertSlice.getResult());
      });
  loop->setAttr("ExtractedLoadOrStore", rewriter.getUnitAttr());

  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

std::optional<RMWOp> translateReduceKind(DescriptorReduceKind kind,
                                         TensorDescType ty) {
  auto scalarTy = ty.getBlockType().getElementType();
  switch (kind) {
  case DescriptorReduceKind::ADD:
    return scalarTy.isInteger() ? RMWOp::ADD : RMWOp::FADD;
  case DescriptorReduceKind::MIN:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMIN;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MIN;
    }
    return {};
  case DescriptorReduceKind::MAX:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMAX;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MAX;
    }
    return {};
  case DescriptorReduceKind::AND:
    return RMWOp::AND;
  case DescriptorReduceKind::OR:
    return RMWOp::OR;
  case DescriptorReduceKind::XOR:
    return RMWOp::XOR;
  default:
    break;
  }
  return {};
}

LogicalResult DescriptorReduceConverter::matchAndRewrite(
    triton::DescriptorReduceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto descTy = op.getDesc().getType();
  const auto blockShape = descTy.getBlockType().getShape();
  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);
  auto offsets = castToI64(rewriter, op.getIndices());
  auto rmwOp = translateReduceKind(op.getKind(), descTy);
  if (!rmwOp) {
    std::string msgstring;
    llvm::raw_string_ostream msg(msgstring);
    msg << "Cannot fallback on descriptor atomic op, unsupported for type "
        << descTy.getBlockType().getElementType();
    return op->emitError(msgstring);
  }

  rewriter.create<triton::AtomicRMWOp>(
      loc, descTy.getSignlessBlockType(), *rmwOp,
      generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
      generateMask(rewriter, loc, blockShape, desc, offsets),
      MemSemantic::ACQUIRE_RELEASE, MemSyncScope::GPU);
  op.erase();
  return success();
}

} // namespace DescriptorConverter
