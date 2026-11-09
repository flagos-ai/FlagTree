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

#ifndef TRITON_ADAPTER_LOADSTORECONVERTER_H
#define TRITON_ADAPTER_LOADSTORECONVERTER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"

namespace LoadStoreConverter {

using namespace mlir;
using namespace triton;

class AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
public:
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Materialize pointer tensor expressions directly instead of routing them
/// through a synthetic AddPtrOp with a zero offset. A higher benefit than the
/// generic value converters ensures pointer layout is handled by
/// BlockDataParser.
template <typename OpTy>
class MemoryPointerConverter : public OpConversionPattern<OpTy> {
public:
  explicit MemoryPointerConverter(MLIRContext *context)
      : OpConversionPattern<OpTy>(context, PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy || !isa<triton::PointerType>(resultTy.getElementType()))
      return failure();

    llvm::SmallDenseMap<Value, BlockData> known;
    FailureOr<Value> memref =
        BlockDataParser::materializePointer(op->getResult(0), rewriter, known);
    if (failed(memref))
      return rewriter.notifyMatchFailure(
          op, "unsupported pointer expression for direct materialization");

    rewriter.replaceOp(op, *memref);
    return success();
  }
};

class LoadConverter : public OpConversionPattern<triton::LoadOp> {
private:
  void propagateWasBoolToInt8Attr(Operation *srcLoadOp, Operation *dstOp,
                                  PatternRewriter &rewriter) const;

  LogicalResult toTensorAndReplace(triton::LoadOp &op,
                                   RankedTensorType &tensorType, Value localMem,
                                   bool mayImplicitTransposeWithLastAxis,
                                   const Location &loc,
                                   ConversionPatternRewriter &rewriter) const;

  LogicalResult checkModifiedByAddPtrConverter(triton::LoadOp &op) const;

  LogicalResult
  continueModifyFromAddPtrConverter(triton::LoadOp &op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const;

  void
  fillTensorWithOtherForMaskScenario(Value other, Value localMem,
                                     ArrayRef<OpFoldResult> maskDim,
                                     ConversionPatternRewriter &rewriter) const;

  LogicalResult
  replaceMaskedLoadWithTensorOther(triton::LoadOp op, Value alloc,
                                   bool mayImplicitTransposeWithLastAxis,
                                   ConversionPatternRewriter &rewriter) const;

public:
  explicit LoadConverter(MLIRContext *context);
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// tempate class's impl must in header file
template <typename OpTy>
class LoadStoreCanonicalizer : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value ptrVal = op.getPtr();

    // A uniform scalar pointer can be broadcast and immediately extracted at
    // a memory boundary. Fold that round-trip here so scalar pointer
    // conversion does not require a tensor-of-pointer materialization.
    if (auto extractOp = ptrVal.getDefiningOp<tensor::ExtractOp>()) {
      if (auto splatOp =
              extractOp.getTensor().getDefiningOp<triton::SplatOp>()) {
        Value scalarPointer = splatOp.getSrc();
        auto pointerType =
            dyn_cast<triton::PointerType>(scalarPointer.getType());
        if (pointerType && !isa<ShapedType>(pointerType.getPointeeType()) &&
            scalarPointer.getType() == ptrVal.getType()) {
          rewriter.modifyOpInPlace(
              op, [&]() { op->replaceUsesOfWith(ptrVal, scalarPointer); });
          return success();
        }
      }
    }

    auto ptrTy = dyn_cast<RankedTensorType>(ptrVal.getType());
    if (!ptrTy || !isa<triton::PointerType>(ptrTy.getElementType()))
      return failure();

    // ReorderBroadcast may turn
    //   addptr(splat(base), splat(offset))
    // into
    //   splat(addptr(base, offset)).
    // Recover the former shape at the memory boundary so AddPtrConverter can
    // keep using the real (non-synthetic) offset as its lowering anchor.
    auto splatOp = ptrVal.getDefiningOp<triton::SplatOp>();
    if (!splatOp)
      return failure();

    auto scalarAddPtr = splatOp.getSrc().getDefiningOp<triton::AddPtrOp>();
    if (!scalarAddPtr)
      return failure();

    Value ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy,
                                                      scalarAddPtr.getPtr());
    auto offsetTy =
        ptrTy.cloneWith(std::nullopt, scalarAddPtr.getOffset().getType());
    Value offsetSplat = rewriter.create<triton::SplatOp>(
        op.getLoc(), offsetTy, scalarAddPtr.getOffset());
    Value addptr = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTy,
                                                     ptrSplat, offsetSplat);

    rewriter.modifyOpInPlace(op,
                             [&]() { op->replaceUsesOfWith(ptrVal, addptr); });
    return success();
  }
};

class ScalarStoreCanonicalizer : public OpRewritePattern<triton::StoreOp> {
public:
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

class StoreConverter : public OpConversionPattern<triton::StoreOp> {
public:
  explicit StoreConverter(MLIRContext *context);

  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ScalarAtomicRMWCanonicalizer
    : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class ScalarAtomicCASCanonicalizer
    : public OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};

class AtomicCASConverter : public OpConversionPattern<triton::AtomicCASOp> {
public:
  explicit AtomicCASConverter(MLIRContext *context)
      : OpConversionPattern<triton::AtomicCASOp>(context) {}
  using OpConversionPattern<triton::AtomicCASOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class AtomicRMWConverter : public OpConversionPattern<triton::AtomicRMWOp> {
private:
  Value createAtomicBinaryOps(OpBuilder &builder, Location loc,
                              triton::AtomicRMWOp op, Type elementType,
                              Value lhs, Value rhs) const {
    auto rmwOp = op.getAtomicRmwOp();

    // it has been confirmed in AtomicRMWConverter::matchAndRewrite
    // that the ptr of op is of MemRefType
    Value binaryOp;
    if (rmwOp == triton::RMWOp::FADD) {
      binaryOp = builder.create<arith::AddFOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::ADD) {
      binaryOp = builder.create<arith::AddIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::XOR) {
      binaryOp = builder.create<arith::XOrIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::OR) {
      binaryOp = builder.create<arith::OrIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::AND) {
      binaryOp = builder.create<arith::AndIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::MAX) {
      // Max/Min only support f32/i32 for now
      // Other type is not supported because of semantic.py
      if (isa<FloatType>(elementType)) {
        binaryOp = builder.create<arith::MaxNumFOp>(loc, lhs, rhs);
      } else {
        binaryOp = builder.create<arith::MaxSIOp>(loc, lhs, rhs);
      }
    } else if (rmwOp == triton::RMWOp::MIN) {
      if (isa<FloatType>(elementType)) {
        binaryOp = builder.create<arith::MinNumFOp>(loc, lhs, rhs);
      } else {
        binaryOp = builder.create<arith::MinSIOp>(loc, lhs, rhs);
      }
    } else if (rmwOp == triton::RMWOp::XCHG) {
      binaryOp = rhs;
    } else if (rmwOp == triton::RMWOp::UMAX) {
      binaryOp = builder.create<arith::MaxUIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::UMIN) {
      binaryOp = builder.create<arith::MinUIOp>(loc, lhs, rhs);
    } else {
      op.emitOpError("unsupported atomic RMW operation: ");
      llvm_unreachable(
          "Not implemented. Support fadd, add, max, min for now !");
    }
    return binaryOp;
  }

  // used when handling scalar
  // to verify whether we need to handle this scalar
  bool isConstantMaskTrue(Value mask) const {
    if (auto denseAttr =
            mask.getDefiningOp()->getAttrOfType<DenseElementsAttr>("value")) {
      auto eleType = denseAttr.getType().getElementType();
      if (isa<IntegerType>(eleType) &&
          cast<IntegerType>(eleType).getWidth() == 1) {
        auto values = denseAttr.getValues<bool>();
        return values[0];
      }
    }
    return false;
  }

  DenseSet<triton::RMWOp> softwareAtomicKinds = {
      triton::RMWOp::AND, triton::RMWOp::OR, triton::RMWOp::XOR};

public:
  explicit AtomicRMWConverter(MLIRContext *context);
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class AtomicMaxMinCanonicalizer : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class ReinterpretCastStrideCanonicalizer
    : public OpRewritePattern<memref::ReinterpretCastOp> {
public:
  using OpRewritePattern<memref::ReinterpretCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op,
                                PatternRewriter &rewriter) const override;
  static bool hasFixableZeroStride(memref::ReinterpretCastOp op);
};

} // namespace LoadStoreConverter
#endif
