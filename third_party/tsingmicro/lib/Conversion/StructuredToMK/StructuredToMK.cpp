//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMK/StructuredToMK.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/StructuredToMK/Passes.h.inc"

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

namespace {

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

void mlir::triton::populateStructuredToMKConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AtomicRMWOpConverter, AtomicCASOpConverter>(
      patterns.getContext());
}
