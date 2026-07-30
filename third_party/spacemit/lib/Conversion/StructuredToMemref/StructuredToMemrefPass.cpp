//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. ALL rights reserved.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights
// reserved. SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STRUCTUREDTOMEMREF
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct ReifyMemrefUnrealizedCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op->getNumResults() != 1)
      return failure();

    auto input = op.getInputs().front();
    auto inputType = dyn_cast<BaseMemRefType>(input.getType());
    auto resultType = dyn_cast<BaseMemRefType>(op.getResult(0).getType());
    if (!inputType || !resultType)
      return failure();

    if (input.getType() == op.getResult(0).getType()) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto rankedInputType = dyn_cast<MemRefType>(input.getType())) {
      if (auto rankedResultType =
              dyn_cast<MemRefType>(op.getResult(0).getType())) {
        if (!memref::CastOp::areCastCompatible(rankedInputType,
                                               rankedResultType))
          return failure();
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, rankedResultType,
                                                    input);
        return success();
      }
      if (auto unrankedResultType =
              dyn_cast<UnrankedMemRefType>(op.getResult(0).getType())) {
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, unrankedResultType,
                                                    input);
        return success();
      }
      return failure();
    }

    if (auto unrankedInputType =
            dyn_cast<UnrankedMemRefType>(input.getType())) {
      if (auto rankedResultType =
              dyn_cast<MemRefType>(op.getResult(0).getType())) {
        if (!memref::CastOp::areCastCompatible(unrankedInputType,
                                               rankedResultType))
          return failure();
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, rankedResultType,
                                                    input);
        return success();
      }
      if (auto unrankedResultType =
              dyn_cast<UnrankedMemRefType>(op.getResult(0).getType())) {
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, unrankedResultType,
                                                    input);
        return success();
      }
    }

    return failure();
  }
};

static ptr::MemorySpaceAttrInterface getPtrBridgeMemorySpace(MLIRContext *ctx) {
  return ptr::GenericSpaceAttr::get(ctx);
}

class PtrToMemrefConverter : public TypeConverter {
public:
  PtrToMemrefConverter() {
    addConversion([](Type type) { return type; });
    // Handle tensor<NxM...x!tt.ptr<elemType>> -> memref<NxM...xelemType>
    // This is for structured pointers like tensor<512x!tt.ptr<i64>>
    addConversion([](RankedTensorType tensorType) -> std::optional<Type> {
      auto ptrType = dyn_cast<triton::PointerType>(tensorType.getElementType());
      if (!ptrType)
        return std::nullopt;

      MLIRContext *ctx = tensorType.getContext();
      Type pointeeType = ptrType.getPointeeType();
      SmallVector<int64_t> shape(tensorType.getShape());

      SmallVector<int64_t> dynStrides(tensorType.getRank(),
                                      ShapedType::kDynamic);
      auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                           dynStrides);

      return MemRefType::get(shape, pointeeType, layout,
                             getPtrBridgeMemorySpace(ctx));
    });
    addConversion([](triton::PointerType ptrType) -> Type {
      MLIRContext *ctx = ptrType.getContext();
      Type pointeeType = ptrType.getPointeeType();

      if (auto tensorType = dyn_cast<RankedTensorType>(pointeeType)) {
        SmallVector<int64_t> shape(tensorType.getShape());
        Type elementType = tensorType.getElementType();

        SmallVector<int64_t> dynStrides(tensorType.getRank(),
                                        ShapedType::kDynamic);
        auto layout = StridedLayoutAttr::get(
            ctx, /*offset=*/ShapedType::kDynamic, dynStrides);

        return MemRefType::get(shape, elementType, layout,
                               getPtrBridgeMemorySpace(ctx));
      }

      SmallVector<int64_t> dynStrides(/*Size=*/1, ShapedType::kDynamic);
      auto layout =
          StridedLayoutAttr::get(ctx,
                                 /*offset=*/ShapedType::kDynamic, dynStrides);
      return MemRefType::get({ShapedType::kDynamic}, pointeeType, layout,
                             getPtrBridgeMemorySpace(ctx));
    });
    addConversion([](xsmt::BufferType bufTy) -> Type {
      MLIRContext *ctx = bufTy.getContext();
      ArrayRef<int64_t> shape = bufTy.getShape();
      Type elemTy = bufTy.getElementType();

      SmallVector<int64_t> dynStrides(shape.size(), ShapedType::kDynamic);
      auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                           dynStrides);

      return MemRefType::get(shape, elemTy, layout, /*memorySpace=*/0);
    });
    addConversion([](xsmt::MBarrierType t) -> Type {
      auto *ctx = t.getContext();
      auto i64 = IntegerType::get(ctx, 64);
      int64_t n = t.getNumBarriers();
      return RankedTensorType::get({n}, i64);
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    });

    // Handle memref type conversions where strides differ (static vs dynamic)
    addTargetMaterialization([&](OpBuilder &builder, MemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      if (isa<triton::PointerType>(inputs[0].getType()))
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs)
            .getResult(0);
      auto inputType = dyn_cast<MemRefType>(inputs[0].getType());
      if (!inputType)
        return nullptr;
      // Check element type and rank match
      if (inputType.getElementType() != resultType.getElementType() ||
          inputType.getRank() != resultType.getRank())
        return nullptr;
      // Let memref::CastOp decide compatibility. It correctly allows
      // static->dynamic (loosening) and other valid memref casts.
      if (!memref::CastOp::areCastCompatible(inputType, resultType))
        return nullptr;

      return memref::CastOp::create(builder, loc, resultType, inputs[0])
          .getResult();
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    });
  }
};

class StructuredToMemrefPass
    : public triton::impl::StructuredToMemrefBase<StructuredToMemrefPass> {
  using StructuredToMemrefBase<StructuredToMemrefPass>::StructuredToMemrefBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, triton::TritonDialect,
                    ttx::TritonTilingExtDialect, memref::MemRefDialect,
                    xsmt::XSMTDialect, xsmt_async::XSMTAsyncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns1(&getContext());
    triton::ViewOpPtrPatternConversionPatterns(patterns1);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns1)))) {
      signalPassFailure();
    }

    RewritePatternSet patterns0(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect, xsmt_async::XSMTAsyncDialect>();

    target.addIllegalOp<tts::LoadOp, tts::StoreOp, tts::MakeTensorPtrOp,
                        xsmt::AllocOp, xsmt::SubviewOp, xsmt::SubviewPackOp,
                        xsmt::MBarrierCopiesOp, xsmt::MBarrierSubviewOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    PtrToMemrefConverter typeConverter;

    triton::populateStructuredToMemrefConversionPatterns(patterns0,
                                                         typeConverter);

    if (failed(
            applyPartialConversion(moduleOp, target, std::move(patterns0)))) {
      signalPassFailure();
    }

    RewritePatternSet cleanupPatterns(&getContext());
    cleanupPatterns.add<ReifyMemrefUnrealizedCast>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(cleanupPatterns)))) {
      signalPassFailure();
    }

    // Change the second input of linalg powop to a scalar
    bool PowTensorToScalar = true;
    if (PowTensorToScalar) {
      SmallVector<linalg::PowFOp> targetOps;
      moduleOp.walk([&](linalg::PowFOp powfOp) {
        // Check if the second operand is a bufferization.to_tensor result
        if (auto toTensor = powfOp.getInputs()[1]
                                .getDefiningOp<bufferization::ToTensorOp>()) {
          // Verify the memref comes from an alloc operation with shape [1]
          if (auto alloc =
                  toTensor.getBuffer().getDefiningOp<memref::AllocOp>()) {
            if (alloc.getType().getShape() == ArrayRef<int64_t>{1}) {
              targetOps.push_back(powfOp);
            }
          }
        }
      });
      for (auto powfOp : targetOps) {
        OpBuilder builder(powfOp);
        auto toTensor =
            powfOp.getInputs()[1].getDefiningOp<bufferization::ToTensorOp>();
        Value memref = toTensor.getBuffer();

        // Create load operation to get the scalar value
        Value loaded =
            memref::LoadOp::create(builder, toTensor.getLoc(), memref,
                                   ValueRange{arith::ConstantIndexOp::create(
                                       builder, toTensor.getLoc(), 0)});

        // Prepare new input list:
        // - Keep first input unchanged
        // - Replace second input with loaded scalar
        SmallVector<Value> newInputs = {powfOp.getInputs()[0], loaded};

        auto newPowf = linalg::PowFOp::create(builder, powfOp.getLoc(),
                                              powfOp.getResultTypes(),
                                              newInputs, powfOp.getOutputs());

        powfOp.getResult(0).replaceAllUsesWith(newPowf.getResult(0));

        powfOp.erase();

        if (toTensor.use_empty()) {
          toTensor.erase();
          if (auto alloc = memref.getDefiningOp<memref::AllocOp>()) {
            if (alloc.use_empty()) {
              alloc.erase();
            }
          }
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass() {
  return std::make_unique<StructuredToMemrefPass>();
}
