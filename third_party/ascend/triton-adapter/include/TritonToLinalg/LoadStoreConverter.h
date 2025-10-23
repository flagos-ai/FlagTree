#ifndef TRITON_ADAPTER_LOADSTORECONVERTER_H
#define TRITON_ADAPTER_LOADSTORECONVERTER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

#include "triton/Dialect/Triton/IR/Dialect.h"

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

class LoadConverter : public OpConversionPattern<triton::LoadOp> {
private:
  LogicalResult toTensorAndReplace(triton::LoadOp &op,
                                   RankedTensorType &tensorType,
                                   memref::AllocOp &allocOp,
                                   bool mayImplicitTransposeWithLastAxis,
                                   const Location &loc,
                                   ConversionPatternRewriter &rewriter) const;

  LogicalResult checkModifiedByAddPtrConverter(triton::LoadOp &op) const;

  LogicalResult
  continueModifyFromAddPtrConverter(triton::LoadOp &op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const;

  void
  fillTensorWithOtherForMaskScenario(Value other, memref::AllocOp localMem,
                                     ArrayRef<OpFoldResult> maskDim,
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
    Type ptrTy = ptrVal.getType();
    auto ptrDefOp = ptrVal.getDefiningOp();
    if (isa<BlockArgument>(ptrVal))
      return failure();

    if (!isTensorPointerType(ptrTy) &&
        !isa_and_nonnull<triton::AddPtrOp>(ptrDefOp)) {
      if (isa<triton::BitcastOp>(ptrDefOp)) {
        auto castOp = cast<triton::BitcastOp>(ptrDefOp);
        auto castSrc = castOp.getSrc();
        if (!isa<BlockArgument>(castSrc)) {
          auto castSrcDefOp = castSrc.getDefiningOp();
          if (isa<triton::AddPtrOp>(castSrcDefOp)) {
            return rewriter.notifyMatchFailure(
                op, "BitcastCanonicalizer handles addptr->bitcast->load!");
          }
        }
      }

      Type zeroTy = getI32SameShape(ptrTy);
      Value zeroVal =
          createScalarOrSplatConstant(rewriter, op.getLoc(), zeroTy, 0);
      Value addptrVal = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTy,
                                                          ptrVal, zeroVal);
      rewriter.modifyOpInPlace(
          op, [&]() { op->replaceUsesOfWith(ptrVal, addptrVal); });
      return success();
    }
    return failure();
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
  explicit AtomicCASConverter(MLIRContext *context) :
    OpConversionPattern<triton::AtomicCASOp>(context) {}
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

} // namespace LoadStoreConverter
#endif