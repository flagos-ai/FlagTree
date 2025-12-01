//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Address/Dialect/IR/AddressDialect.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "utils/TypeConvertor.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Casting.h"

#include <optional>

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

class LoopTypeConverter : public TypeConverter {
public:
  LoopTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });

    // A tensor of pointers can be passed in as scf.for's init-args, in such
    // cases, we convert the type to a memref with dynamic offsets and
    // strides.
    addConversion(
        [context](RankedTensorType tensorType) -> std::optional<MemRefType> {
          if (auto ptrType = llvm::dyn_cast<triton::PointerType>(
                  tensorType.getElementType())) {
            auto layout = StridedLayoutAttr::get(
                context, ShapedType::kDynamic,
                SmallVector<int64_t>(tensorType.getRank(),
                                     ShapedType::kDynamic));
            Type elemType = ptrType.getPointeeType();
            return MemRefType::get(tensorType.getShape(), elemType, layout);
          }

          return std::nullopt;
        });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    // Convert the current memref type to a memref type with dynamic offsets and
    // strides through another reinterpret_cast with the same offsets.
    // Canonicalization will simplify this sequence by removing the inital
    // reinterpret_cast.
    addTargetMaterialization([&](OpBuilder &builder, MemRefType memrefType,
                                 ValueRange inputs, Location loc) -> Value {
      auto reinterpretCast =
          inputs[0].getDefiningOp<memref::ReinterpretCastOp>();
      if (!reinterpretCast) {
        return builder
            .create<UnrealizedConversionCastOp>(loc, memrefType, inputs)
            .getResult(0);
      }
      return builder.create<memref::ReinterpretCastOp>(
          loc, memrefType, inputs[0], reinterpretCast.getMixedOffsets()[0],
          reinterpretCast.getMixedSizes(), reinterpretCast.getMixedStrides());
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
                    ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect, addr::AddressDialect, mk::MagicKernelDialect>();

    target.addIllegalOp<tts::LoadOp, tts::StoreOp, tts::MakeTensorPtrOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    LoopTypeConverter loopTypeConverter(patterns.getContext());

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        loopTypeConverter, patterns, target);

    PtrToUnrankedMemrefConverter typeConverter;
    triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                         typeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass() {
  return std::make_unique<StructuredToMemrefPass>();
}
