//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "unstructured-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_UNSTRUCTUREDTOMEMREF
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"
} // namespace mlir::triton

namespace {

static ptr::MemorySpaceAttrInterface getPtrBridgeMemorySpace(MLIRContext *ctx) {
  return ptr::GenericSpaceAttr::get(ctx);
}

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      auto *ctx = ptrType.getContext();
      return UnrankedMemRefType::get(ptrType.getPointeeType(),
                                     getPtrBridgeMemorySpace(ctx));
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    });
  }
};

static MemRefType getMemrefTypeForScalarPtr(triton::PointerType ptrType,
                                            MLIRContext *context) {
  SmallVector<int64_t> strides{1};
  auto layout = StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);
  auto elemType = ptrType.getPointeeType();
  auto memrefType =
      MemRefType::get({1}, elemType, layout, getPtrBridgeMemorySpace(context));
  return memrefType;
}

struct ScalarLoadConverter : public OpConversionPattern<tts::GatherOp> {
  using OpConversionPattern<tts::GatherOp>::OpConversionPattern;

  ScalarLoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(typeConverter, context) {}

  ScalarLoadConverter(MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!gatherOp.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = gatherOp->getLoc();

    auto basePtr = adaptor.getPtr();
    auto offset = adaptor.getOffset();

    Value loadIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), offset);

    auto memref = memref::ReinterpretCastOp::create(
        rewriter, loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(gatherOp.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(loadIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    auto scalarLoadOp = affine::AffineLoadOp::create(rewriter, loc, memref,
                                                     zeroMap, ValueRange{});

    rewriter.replaceOp(gatherOp, scalarLoadOp.getResult());

    return success();
  }
};

struct ScalarStoreConverter : public OpConversionPattern<tts::ScatterOp> {
  using OpConversionPattern<tts::ScatterOp>::OpConversionPattern;

  ScalarStoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(typeConverter, context) {}

  ScalarStoreConverter(MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!scatterOp.getValue().getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = scatterOp->getLoc();

    auto basePtr = adaptor.getPtr();
    auto offset = adaptor.getOffset();

    Value storeIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), offset);

    auto memref = memref::ReinterpretCastOp::create(
        rewriter, loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(scatterOp.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(storeIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto storeVal = scatterOp.getValue();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    affine::AffineStoreOp::create(rewriter, loc, storeVal, memref, zeroMap,
                                  ValueRange{});
    rewriter.eraseOp(scatterOp);

    return success();
  }
};

// Lowering an unstructured load op (gather) into a linalg.generic op.
struct GatherConverter : public OpConversionPattern<tts::GatherOp> {
  using OpConversionPattern<tts::GatherOp>::OpConversionPattern;

  GatherConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(typeConverter, context) {}

  GatherConverter(MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gatherOp->getLoc();

    auto ptr = adaptor.getPtr();
    auto offsetTensor = adaptor.getOffset();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    // This must be a scalar load, skip processing.
    if (!offsetType) {
      return failure();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(gatherOp.getResult().getType());

    // Treat the base pointer (memref) as 1D because the offsets are all
    // relative to a single base pointer (already collapsed).
    auto baseMemref =
        memref::CastOp::create(
            rewriter, loc,
            MemRefType::get({ShapedType::kDynamic}, resultType.getElementType(),
                            AffineMap(),
                            getPtrBridgeMemorySpace(rewriter.getContext())),
            ptr)
            .getResult();

    auto baseTensor =
        bufferization::ToTensorOp::create(
            rewriter, loc,
            RankedTensorType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                                  resultType.getElementType()),
            baseMemref, true /* restrict */, false /* writable */)
            .getResult();

    // The linalg.generic op should have the following inputs:
    // - the offset tensor.
    // - an optional mask tensor if the gather op contains mask.
    SmallVector<Value> inputs{offsetTensor};

    if (gatherOp.getMask()) {
      inputs.push_back(gatherOp.getMask());
    }

    auto emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                resultType.getElementType())
            .getResult();

    // Affine maps for the inputs and one additional output.
    SmallVector<AffineMap> affineMaps(
        inputs.size() + 1,
        rewriter.getMultiDimIdentityMap(resultType.getRank()));

    // All iterator types are parallel.
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, inputs, ValueRange{emptyTensor},
        affineMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto getValueAtIndex = [baseTensor](OpBuilder &b, Location loc,
                                              Value index) -> Value {
            Value index0 =
                arith::IndexCastOp::create(b, loc, b.getIndexType(), index);

            return tensor::ExtractOp::create(b, loc, baseTensor,
                                             ValueRange{index0});
          };

          auto offset = args[0];

          if (!gatherOp.getMask()) {
            // If there is no mask, simply extract the current element from the
            // base tensor and use it as the yield value.
            auto loadValue = getValueAtIndex(b, loc, offset);
            linalg::YieldOp::create(b, loc, loadValue);
          } else {
            // If the mask value is truthy, the current element is loaded from
            // the base tensor using its offset. Otherwise, if `other` is
            // present, yield `other`. If `other` is not present, a default
            // value of 0 is used.
            auto mask = args[1];
            auto ifOp = scf::IfOp::create(
                b, loc, mask,
                [&](OpBuilder &b, Location loc) {
                  // Truthy case, load from the index.
                  auto value = getValueAtIndex(b, loc, offset);
                  scf::YieldOp::create(b, loc, value);
                },
                [&](OpBuilder &b, Location loc) {
                  // Falsy case, yield `other` or 0 as the default value.
                  if (gatherOp.getOther()) {
                    scf::YieldOp::create(b, loc, gatherOp.getOther());
                  } else {
                    auto elemType = resultType.getElementType();
                    auto zeroAttr = b.getZeroAttr(elemType);
                    assert(zeroAttr && "unexpected element type");
                    Value extract = arith::ConstantOp::create(b, loc, zeroAttr);
                    scf::YieldOp::create(b, loc, extract);
                  }
                });

            linalg::YieldOp::create(b, loc, ifOp->getResult(0));
          }
        });

    rewriter.replaceOp(gatherOp, genericOp);

    return success();
  }
};

// Lowering an unstructured store op (scatter) into a linalg.generic op.
struct ScatterConverter : public OpConversionPattern<tts::ScatterOp> {
  using OpConversionPattern<tts::ScatterOp>::OpConversionPattern;

  ScatterConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(typeConverter, context) {}

  ScatterConverter(MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = scatterOp->getLoc();

    auto ptr = adaptor.getPtr();
    auto offsetTensor = adaptor.getOffset();
    auto valueTensor = adaptor.getValue();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    // This must be a scalar store, skip processing.
    if (!offsetType) {
      return failure();
    }

    auto valueType = dyn_cast<RankedTensorType>(scatterOp.getValue().getType());

    // Treat the base pointer (memref) as 1D because the offsets are all
    // relative to a single base pointer (already collapsed).
    auto baseMemref =
        memref::CastOp::create(
            rewriter, loc,
            MemRefType::get({ShapedType::kDynamic}, valueType.getElementType(),
                            AffineMap(),
                            getPtrBridgeMemorySpace(rewriter.getContext())),
            ptr)
            .getResult();

    // The linalg.generic op should have the following inputs:
    // - the offset tensor.
    // - the value tensor.
    // - an optional mask tensor if the scatter op contains mask.
    SmallVector<Value> inputs{offsetTensor, valueTensor};

    if (scatterOp.getMask()) {
      inputs.push_back(scatterOp.getMask());
    }

    // Affine maps for the inputs.
    SmallVector<AffineMap> affineMaps(
        inputs.size(), rewriter.getMultiDimIdentityMap(valueType.getRank()));

    // All iterator types are parallel.
    SmallVector<utils::IteratorType> iteratorTypes(
        valueType.getRank(), utils::IteratorType::parallel);

    rewriter.setInsertionPoint(scatterOp);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{}, inputs, ValueRange{}, affineMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          auto storeValueAtIndex = [baseMemref](OpBuilder &b, Location loc,
                                                Value index, Value value) {
            Value index0 =
                arith::IndexCastOp::create(b, loc, b.getIndexType(), index);

            memref::StoreOp::create(b, loc, value, baseMemref,
                                    ValueRange{index0});
          };

          auto offset = args[0];
          auto value = args[1];

          if (!scatterOp.getMask()) {
            // If there is no mask, simply insert the current value to the
            // base memref using its offset.
            storeValueAtIndex(b, loc, offset, value);
          } else {
            // If the mask value is truthy, insert the current value to the
            // the base memref using its offset. Otherwise, noop.
            auto mask = args[2];
            auto ifOp = scf::IfOp::create(
                b, loc, mask, [&](OpBuilder &b, Location loc) {
                  storeValueAtIndex(b, loc, offset, value);
                  scf::YieldOp::create(b, loc);
                });
          }

          linalg::YieldOp::create(b, loc);
        });

    rewriter.eraseOp(scatterOp);

    return success();
  }
};

class UnstructuredToMemrefPass
    : public triton::impl::UnstructuredToMemrefBase<UnstructuredToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect>();

    target.addIllegalOp<tts::GatherOp, tts::ScatterOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<GatherConverter, ScatterConverter, ScalarLoadConverter,
                 ScalarStoreConverter>(typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createUnstructuredToMemrefPass() {
  return std::make_unique<UnstructuredToMemrefPass>();
}
