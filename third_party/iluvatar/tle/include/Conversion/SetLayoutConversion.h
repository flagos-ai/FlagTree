// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#ifndef ILUVATAR_TLE_SET_LAYOUT_CONVERSION_H
#define ILUVATAR_TLE_SET_LAYOUT_CONVERSION_H

#include <optional>
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "Transforms/ExplicitEncoding.h"

namespace mlir::triton::iluvatar_tle {

inline void legalizeElementwiseEncodings(Operation *op,
                                        SmallVector<Type> &resultTypes,
                                        SmallVector<Value> &operands,
                                        ConversionPatternRewriter &rewriter) {
  if (!op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() &&
      !op->hasTrait<OpTrait::Elementwise>())
    return;
  Attribute encoding;
  for (auto [oldValue, newValue] : llvm::zip(op->getOperands(), operands)) {
    if (oldValue.getDefiningOp<tle::SetLayoutOp>()) {
      encoding = cast<RankedTensorType>(newValue.getType()).getEncoding();
      break;
    }
  }
  if (!encoding)
    for (Type type : resultTypes)
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        encoding = tensor.getEncoding();
        break;
      }
  if (!encoding)
    return;
  for (Type &type : resultTypes)
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.cloneWithEncoding(encoding);
  for (Value &value : operands)
    if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
      if (tensor.getEncoding() != encoding)
        value = rewriter.create<gpu::ConvertLayoutOp>(
            op->getLoc(), tensor.cloneWithEncoding(encoding), value);
}

// The shared Python API emits TleDialect hints even when the surrounding
// backend uses IluvatarTleDialect. Convert the hint at the TTIR type boundary;
// leaving it legal would retain an unencoded operand after load conversion.
class SetLayoutConversion : public OpConversionPattern<tle::SetLayoutOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::SetLayoutOp op, tle::SetLayoutOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto original = cast<RankedTensorType>(op.getType());
    auto resultType = original.cloneWithEncoding(op.getTargetEncoding());
    Value source = adaptor.getSrc();
    // An adaptor may be a temporary type-conversion materialization. Its
    // producer can disappear when conversion mappings are reconciled, so
    // attach the contract to a dedicated replacement, including identity
    // conversions, rather than mutating that temporary producer.
    auto convert = rewriter.replaceOpWithNewOp<gpu::ConvertLayoutOp>(
        op, resultType, source);
    convert->setAttr("tle.explicit_encoding.0", op.getTargetEncoding());
    return success();
  }
};

class ReshapeConversion : public OpConversionPattern<ReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ReshapeOp op, ReshapeOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Reshape has an encoding inferred from its adapted source. Assigning
    // the module default to its result contradicts a preceding layout hint.
    auto attributes = op->getAttrs();
    auto reshaped = rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, op.getType().getShape(), adaptor.getSrc(), op.getAllowReorder());
    reshaped->setAttrs(attributes);
    return success();
  }
};

class ForConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, scf::ForOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // A loop's init, region argument, yield and result are one type contract.
    // Independently choosing the default for region arguments contradicts an
    // explicitly encoded initializer even before layout optimization runs.
    TypeConverter::SignatureConversion signature(op.getBody()->getNumArguments());
    signature.addInputs(0, op.getInductionVar().getType());
    for (auto [index, value] : llvm::enumerate(adaptor.getInitArgs()))
      signature.addInputs(index + 1, value.getType());
    auto converted = cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op));
    rewriter.inlineRegionBefore(op.getRegion(), converted.getRegion(),
                                converted.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&converted.getRegion(),
                                          *getTypeConverter(), &signature)))
      return rewriter.notifyMatchFailure(op, "could not convert loop body");
    converted->setOperands(adaptor.getOperands());
    for (auto [result, value] : llvm::zip(converted.getResults(), adaptor.getInitArgs()))
      result.setType(value.getType());
    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

class YieldConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, scf::YieldOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> values(adaptor.getOperands());
    Operation *parent = op->getParentOp();
    for (auto [value, type] : llvm::zip(values, parent->getResultTypes())) {
      if (value.getType() == type)
        continue;
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor || !tensor.getEncoding())
        return failure();
      value = rewriter.create<gpu::ConvertLayoutOp>(op.getLoc(), tensor, value);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, values);
    return success();
  }
};

class ReturnConversion : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto function = op->getParentOfType<triton::FuncOp>();
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(function.getResultTypes(), resultTypes)))
      return failure();
    SmallVector<Value> values(adaptor.getOperands());
    for (auto [index, type] : llvm::enumerate(resultTypes)) {
      if (values[index].getType() == type)
        continue;
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor || !tensor.getEncoding())
        return failure();
      values[index] = rewriter.create<gpu::ConvertLayoutOp>(
          op.getLoc(), tensor, values[index]);
    }
    rewriter.replaceOpWithNewOp<triton::ReturnOp>(op, values);
    return success();
  }
};

inline void populateSetLayoutConversion(TypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       ConversionTarget &target) {
  // Operand adaptors ask the converter for each original SSA value's type.
  // The hint's unencoded result must select its requested layout here too;
  // otherwise adaptors immediately convert it back to the default encoding.
  converter.addConversion([](Value value) -> std::optional<Type> {
    if (auto hint = value.getDefiningOp<tle::SetLayoutOp>())
      return cast<RankedTensorType>(value.getType())
          .cloneWithEncoding(hint.getTargetEncoding());
    return std::nullopt;
  });
  target.addIllegalOp<tle::SetLayoutOp>();
  patterns.add<SetLayoutConversion>(converter, patterns.getContext());
  patterns.add<ReshapeConversion>(converter, patterns.getContext(), PatternBenefit(10));
  patterns.add<ForConversion, YieldConversion>(converter, patterns.getContext(),
                                              PatternBenefit(10));
  patterns.add<ReturnConversion>(converter, patterns.getContext(), PatternBenefit(10));
}

// Already-GPU async copies may be emitted by TLE in otherwise unencoded
// TTIR. Their pointers and masks still need the same operand conversion as
// ordinary Triton memory operations; dialect-wide legality skips that work.
class AsyncCopyConversion
    : public OpConversionPattern<gpu::AsyncCopyGlobalToLocalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::AsyncCopyGlobalToLocalOp op,
                  gpu::AsyncCopyGlobalToLocalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Attribute encoding =
        cast<RankedTensorType>(adaptor.getSrc().getType()).getEncoding();
    SmallVector<Value> operands(adaptor.getOperands());
    for (Value &value : operands)
      if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
        if (tensor.getEncoding() != encoding)
          value = rewriter.create<gpu::ConvertLayoutOp>(
              op.getLoc(), tensor.cloneWithEncoding(encoding), value);
    rewriter.replaceOpWithNewOp<gpu::AsyncCopyGlobalToLocalOp>(
        op, op->getResultTypes(), operands, op->getAttrs());
    return success();
  }
};

inline void populateAsyncCopyConversion(TypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       ConversionTarget &target) {
  target.addDynamicallyLegalOp<gpu::AsyncCopyGlobalToLocalOp>(
      [&](gpu::AsyncCopyGlobalToLocalOp op) {
        if (!converter.isLegal(op.getOperation()))
          return false;
        Attribute encoding =
            cast<RankedTensorType>(op.getSrc().getType()).getEncoding();
        for (Value value : SmallVector<Value>{op.getMask(), op.getOther()})
          if (value && cast<RankedTensorType>(value.getType()).getEncoding() !=
                           encoding)
            return false;
        return true;
      });
  patterns.add<AsyncCopyConversion>(converter, patterns.getContext());
}

template <typename Op>
class MemoryAccessConversion : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pointerType = dyn_cast<RankedTensorType>(adaptor.getPtr().getType());
    if (!pointerType)
      return rewriter.notifyMatchFailure(op, "expected tensor pointers");
    Attribute encoding = pointerType.getEncoding();
    SmallVector<Value> operands;
    for (Value value : adaptor.getOperands()) {
      if (auto type = dyn_cast<RankedTensorType>(value.getType()))
        if (type.getEncoding() != encoding)
          value = rewriter.create<gpu::ConvertLayoutOp>(
              op.getLoc(), type.cloneWithEncoding(encoding), value);
      operands.push_back(value);
    }
    SmallVector<Type> types;
    for (Type type : op->getResultTypes())
      types.push_back(cast<RankedTensorType>(type).cloneWithEncoding(encoding));
    OperationState state(op.getLoc(), Op::getOperationName());
    state.addOperands(operands);
    state.addTypes(types);
    state.addAttributes(op->getAttrs());
    Operation *converted = rewriter.create(state);
    if (Attribute contract = explicitEncoding(adaptor.getPtr()))
      converted->setAttr("tle.explicit_memory_encoding", contract);
    rewriter.replaceOp(op, converted->getResults());
    return success();
  }
};

inline void populateMemoryAccessConversion(TypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target) {
  target.addDynamicallyLegalOp<LoadOp, StoreOp>([&](Operation *op) {
    if (!converter.isLegal(op))
      return false;
    Attribute encoding;
    for (Type type : llvm::concat<Type>(op->getOperandTypes(),
                                       op->getResultTypes())) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (encoding && encoding != tensor.getEncoding())
          return false;
        encoding = tensor.getEncoding();
      }
    }
    return true;
  });
  patterns.add<MemoryAccessConversion<LoadOp>, MemoryAccessConversion<StoreOp>>(
      converter, patterns.getContext(), PatternBenefit(10));
}

} // namespace mlir::triton::iluvatar_tle
#endif
