//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// This pass lowers all triton ops on pointer to their equivalent form in the
// proposed Pointer Dialect:
// https://discourse.llvm.org/t/rfc-ptr-dialect-modularizing-ptr-ops-in-the-llvm-dialect/75142
//
// This pass is intended to be used after all running
// triton-arith-to-linalg="tensor-ptr-to-linalg=true".
// All triton ops on tensors of pointers are expected to have been lowered to
// linalg ops, and that only triton ops on single pointers remain.
//
// Implementation notes:
// Because triton pointers are typed whereas the !ptr.ptr type isn't. The
// lowering for addptr will have to manually scale the offsets by pointee type.
// As a result, bitcasts are no-op after this pass.
//===----------------------------------------------------------------------===//

#include "Address/Dialect/IR/AddressDialect.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/ConvertTritonPtr/TritonPtrToAddress.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "utils/utils.h"

#define DEBUG_TYPE "triton-to-ptr"

using namespace mlir;

namespace {

#define GEN_PASS_DEF_TRITONPTRTOADDRESS
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h.inc"

// arith.select could operate on triton pointers. Convert to use !ptr.ptr
struct SelectOpConverter : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  SelectOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, getTypeConverter()->convertType(op.getType()),
        adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

// Convert bitcast which is a no-op because !ptr.ptr is opaque with no pointtee
// type.
struct BitCastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  BitCastConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::BitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }

    // If the source is a triton pointer, we can convert it to an address
    // type.
    rewriter.replaceOpWithNewOp<mk::BitcastOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.getSrc());
    return success();
  }
};

// Convert tt.ptr_to_int to ptr.ptrtoint
struct PtrToIntConverter : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern<triton::PtrToIntOp>::OpConversionPattern;

  PtrToIntConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::PtrToIntOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<addr::CastIntOp>(op, op.getType(),
                                                 adaptor.getSrc());
    return success();
  }
};

// Convert tt.int_to_ptr to ptr.ptrtoint
struct IntToPtrConverter : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern<triton::IntToPtrOp>::OpConversionPattern;

  IntToPtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::IntToPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<addr::CastIntOp>(
        op, addr::AddressType::get(rewriter.getContext()), adaptor.getSrc());
    return success();
  }
};

class TritonPtrTypeConverter : public TypeConverter {
public:
  TritonPtrTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      return addr::AddressType::get(context);
    });
    addConversion([context](RankedTensorType tensorType) {
      if (isa<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(tensorType.getShape(),
                                     addr::AddressType::get(context));
      }
      return tensorType;
    });
    auto createCast = [&](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addTargetMaterialization(createCast);
    addSourceMaterialization(createCast);
  }
};

class TritonPtrToAddressPass
    : public impl::TritonPtrToAddressBase<TritonPtrToAddressPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, addr::AddressDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonPtrTypeConverter typeConverter(&getContext());
    target.addLegalDialect<addr::AddressDialect>();

    target.addIllegalOp<triton::IntToPtrOp, triton::PtrToIntOp>();
    target.addDynamicallyLegalOp<arith::SelectOp>([](auto op) {
      return llvm::all_of(
          llvm::concat<Value>(op->getOperands(), op->getResults()),
          [&](Value v) {
            return !mlir::triton::utils::isPtrTypeLike(v.getType());
          });
    });

    patterns.add<PtrToIntConverter, IntToPtrConverter, SelectOpConverter,
                 BitCastConverter>(typeConverter, patterns.getContext());

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonPtrToAddressPass() {
  return std::make_unique<TritonPtrToAddressPass>();
}
