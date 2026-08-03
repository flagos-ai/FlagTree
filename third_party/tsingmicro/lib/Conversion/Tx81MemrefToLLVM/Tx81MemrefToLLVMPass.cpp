//===------------------- Tx81MemrefToLLVMPass.cpp--------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Tx81MemrefToLLVM.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "tx81-memref-to-llvm"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Pre-processing: Convert unrealized_conversion_cast between memref types
// to legal memref.cast ops, so they can be properly handled by the
// MemRefToLLVM lowering instead of surviving as unrealized casts.
//===----------------------------------------------------------------------===//
//
// memref.cast requires source and target element types to be the same.
// When element types differ (e.g. memref<*xi1> -> memref<*xi8>),
// memref.cast is illegal, so we skip those cases and leave the
// unrealized_conversion_cast for later passes (e.g. Tx81ToLLVM) to
// eliminate naturally when all memrefs become LLVM pointers.
//
struct MemrefUnrealizedCastToMemrefCast : public RewritePattern {
  MemrefUnrealizedCastToMemrefCast(MLIRContext *ctx)
      : RewritePattern(UnrealizedConversionCastOp::getOperationName(), 1,
                       ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  PatternRewriter &rewriter) const override {
    auto castOp = cast<UnrealizedConversionCastOp>(op);
    Value operand = castOp.getOperand(0);
    Type srcType = operand.getType();
    Type dstType = castOp.getResult(0).getType();

    // Only handle casts between two memref types.
    auto srcMemRef = dyn_cast<BaseMemRefType>(srcType);
    auto dstMemRef = dyn_cast<BaseMemRefType>(dstType);
    if (!srcMemRef || !dstMemRef)
      return failure();

    // Same type → just erase the cast.
    if (srcType == dstType) {
      rewriter.replaceOp(op, operand);
      return success();
    }

    // Same element type → use a proper memref.cast (legal for memref.cast).
    if (srcMemRef.getElementType() == dstMemRef.getElementType()) {
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, dstType, operand);
      return success();
    }

    // Different element types — memref.cast is illegal here.
    // Leave the unrealized_conversion_cast in place for later passes
    // (e.g. ReconcileUnrealizedCastsPass) to handle naturally when all memrefs become LLVM
    // pointers and the element type distinction is erased.
    LLVM_DEBUG({
      llvm::dbgs() << "  Skipping unrealized_conversion_cast from "
                   << srcType << " to " << dstType
                   << " (element types differ, memref.cast would be illegal)\n";
    });
    return failure();
  }
};

class Tx81MemrefToLLVMPass
    : public mlir::triton::Tx81MemrefToLLVMBase<Tx81MemrefToLLVMPass> {
  using Tx81MemrefToLLVMBase<Tx81MemrefToLLVMPass>::Tx81MemrefToLLVMBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, tx::Tx81Dialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();

    // Pre-processing: convert unrealized_conversion_cast between memref types
    // to legal memref.cast ops. This must happen before the partial conversion
    // below so that the MemRefCastOpLowering pattern can handle them.
    {
      RewritePatternSet prePatterns(context);
      prePatterns.add<MemrefUnrealizedCastToMemrefCast>(context);
      if (failed(
              applyPatternsGreedily(moduleOp, std::move(prePatterns)))) {
        return signalPassFailure();
      }
    }

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addIllegalOp<
        memref::AllocOp, memref::LoadOp, memref::StoreOp,
        memref::ReinterpretCastOp, memref::ExtractStridedMetadataOp,
        memref::ExtractAlignedPointerAsIndexOp, memref::CastOp>();

    target.addLegalDialect<LLVM::LLVMDialect, memref::MemRefDialect,
                           func::FuncDialect, arith::ArithDialect,
                           math::MathDialect, arith::ArithDialect,
                           affine::AffineDialect, scf::SCFDialect,
                           cf::ControlFlowDialect, tensor::TensorDialect>();

    target.addLegalOp<ModuleOp>();

    LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = false;
    LLVMTypeConverter llvmTypeConverter(context, options);
    triton::populateTx81MemrefToLLVMConversionPatterns(patterns,
                                                       llvmTypeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTx81MemrefToLLVMPass() {
  return std::make_unique<Tx81MemrefToLLVMPass>();
}
