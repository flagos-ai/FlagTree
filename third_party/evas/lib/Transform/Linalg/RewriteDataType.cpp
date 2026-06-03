
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "evas/Transform/Linalg/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#define GEN_PASS_DEF_REWRITEDATATYPE
#include "evas/Transform/Linalg/Passes.h.inc"

namespace mlir::triton::ev {
using namespace mlir;

#define DEBUG_TYPE "ev-rewrite-data-type"

namespace {

bool isScalar(Value val) {
  return !isa<BaseMemRefType, TensorType, triton::PointerType>(val.getType());
}

bool isScalarOp(Operation *op) {
  auto isScalarValue = [](Value val) { return isScalar(val); };
  return llvm::all_of(op->getOperands(), isScalarValue) &&
         llvm::all_of(op->getResults(), isScalarValue);
}

bool isInsideLinalgRegion(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto *dialect = parent->getDialect();
        dialect && dialect->getNamespace() == "linalg")
      return true;
  }
  return false;
}

/// Check if all i64 values in dense attr are within i32 range
bool allValuesInI32Range(DenseIntElementsAttr attr) {
  if (!attr.getElementType().isInteger(64))
    return false;
  return llvm::all_of(attr.getValues<APInt>(), [](const APInt &val) {
    int64_t v = val.getSExtValue();
    return v >= INT32_MIN && v <= INT32_MAX;
  });
}

bool isI64ConstantInI32Range(arith::ConstantOp op) {
  auto value = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    if (intAttr.getType().isInteger(64)) {
      int64_t val = intAttr.getValue().getSExtValue();
      return val >= INT32_MIN && val <= INT32_MAX;
    }
  }
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(value))
    return allValuesInI32Range(denseAttr);
  return false;
}

/// Create integer cast operation based on width comparison
Value createIntegerCast(OpBuilder &builder, Location loc, Type resultType,
                        Value input, bool useSignedExt) {
  auto srcIntType = dyn_cast<IntegerType>(input.getType());
  auto dstIntType = dyn_cast<IntegerType>(resultType);
  if (!srcIntType || !dstIntType)
    return nullptr;

  unsigned srcWidth = srcIntType.getWidth();
  unsigned dstWidth = dstIntType.getWidth();
  if (srcWidth > dstWidth)
    return builder.create<arith::TruncIOp>(loc, resultType, input).getResult();
  if (srcWidth < dstWidth) {
    return useSignedExt ? builder.create<arith::ExtSIOp>(loc, resultType, input)
                              .getResult()
                        : builder.create<arith::ExtUIOp>(loc, resultType, input)
                              .getResult();
  }
  return nullptr;
}

/// Extract integer values from dense attr and convert to target type
template <typename T> SmallVector<T> extractValues(DenseIntElementsAttr attr) {
  SmallVector<T> result;
  for (const APInt &val : attr.getValues<APInt>())
    result.push_back(static_cast<T>(val.getSExtValue()));
  return result;
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

/// Converts i1 -> i8 and i64 -> i32, including nested types in tensors/memrefs
class DataTypeConverter : public TypeConverter {
public:
  DataTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([](IntegerType intType) -> Type {
      unsigned width = intType.getWidth();
      if (width == 1)
        return IntegerType::get(intType.getContext(), 8);
      if (width == 64)
        return IntegerType::get(intType.getContext(), 32);
      return intType;
    });

    addConversion([this](TensorType type) -> Type {
      return convertElementType(type,
                                [](auto t, auto e) { return t.clone(e); });
    });

    addConversion([this](BaseMemRefType type) -> Type {
      return convertElementType(
          type, [](auto t, auto e) { return t.cloneWith(std::nullopt, e); });
    });

    addConversion([this](MemRefType type) -> Type {
      return convertElementType(
          type, [](auto t, auto e) { return t.cloneWith(t.getShape(), e); });
    });

    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) {
      return materialize(builder, resultType, inputs, loc, true);
    });

    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) {
      return materialize(builder, resultType, inputs, loc, true);
    });
  }

private:
  template <typename T, typename CloneFn>
  Type convertElementType(T type, CloneFn cloneFn) {
    Type elemType = type.getElementType();
    Type converted = convertType(elemType);
    return (converted != elemType) ? cloneFn(type, converted) : type;
  }

  static Value materialize(OpBuilder &builder, Type resultType,
                           ValueRange inputs, Location loc, bool useSignedExt) {
    if (inputs.size() != 1)
      return nullptr;
    if (Value cast = createIntegerCast(builder, loc, resultType, inputs[0],
                                       useSignedExt)) {
      return cast;
    }
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Converts arith.constant op - handles both value attribute and result type
struct ConstantOpConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(resultType);
    if (convertedType == resultType)
      return failure();

    Attribute value = op.getValue();

    // Handle dense tensor constants
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(value)) {
      auto newAttr = convertDenseAttr(denseAttr, convertedType);
      if (!newAttr)
        return failure();
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, convertedType,
                                                     newAttr);
      return success();
    }

    // Handle scalar integer constants (i64 -> i32)
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      auto newAttr =
          IntegerAttr::get(convertedType, intAttr.getValue().trunc(32));
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, convertedType,
                                                     newAttr);
      return success();
    }

    return failure();
  }

private:
  /// Convert dense integer attr to new shaped type
  static DenseIntElementsAttr convertDenseAttr(DenseIntElementsAttr attr,
                                               Type convertedType) {
    auto shapedType = cast<ShapedType>(convertedType);
    Type elemType = attr.getElementType();

    // i64 -> i32 conversion
    if (elemType.isInteger(64)) {
      auto values = extractValues<int32_t>(attr);
      return DenseIntElementsAttr::get(shapedType, values);
    }
    // i1 -> i8 conversion
    if (elemType.isInteger(1)) {
      auto values = extractValues<int8_t>(attr);
      return DenseIntElementsAttr::get(shapedType, values);
    }
    return nullptr;
  }
};

/// Generic converter for all ops - clones op with converted types
struct GenericOpConverter : public ConversionPattern {
  GenericOpConverter(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convertedTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                convertedTypes)))
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (size_t i = 0; i < op->getNumRegions(); ++i)
      state.addRegion();

    Operation *newOp = rewriter.create(state);
    inlineRegionsAndConvertBlockArgs(op, newOp, rewriter);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  void
  inlineRegionsAndConvertBlockArgs(Operation *srcOp, Operation *dstOp,
                                   ConversionPatternRewriter &rewriter) const {
    for (auto [src, dst] :
         llvm::zip(srcOp->getRegions(), dstOp->getRegions())) {
      rewriter.inlineRegionBefore(src, dst, dst.end());
      for (Block &block : dst) {
        for (BlockArgument arg : block.getArguments()) {
          Type converted = getTypeConverter()->convertType(arg.getType());
          if (converted != arg.getType())
            arg.setType(converted);
        }
      }
    }
  }
};

class RewriteDataTypePass
    : public ::impl::RewriteDataTypeBase<RewriteDataTypePass> {
  using RewriteDataTypeBase<RewriteDataTypePass>::RewriteDataTypeBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    DataTypeConverter typeConverter;

    ConversionTarget target(*context);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      // arith.constant with i64 values in i32 range should be converted
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (isI64ConstantInI32Range(constOp)) {
          return typeConverter.isLegal(op->getResultTypes());
        }
      }

      // Scalar ops outside linalg regions are legal (not converted)
      if (isScalarOp(op) && !isInsideLinalgRegion(op)) {
        return true;
      }
      return typeConverter.isLegal(op->getResultTypes()) &&
             typeConverter.isLegal(op->getOperandTypes());
    });

    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    patterns.add<GenericOpConverter>(typeConverter, context);
    patterns.add<ConstantOpConverter>(typeConverter, context);

    if (failed(applyPartialConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Run canonicalizer to clean up unrealized conversion casts
    mlir::PassManager pm(m.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    (void)pm.run(m);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRewriteDataTypePass() {
  return std::make_unique<RewriteDataTypePass>();
}

} // namespace mlir::triton::ev

