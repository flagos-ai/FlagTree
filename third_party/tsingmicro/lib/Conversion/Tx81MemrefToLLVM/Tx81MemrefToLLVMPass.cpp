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
#include "mlir/IR/BuiltinTypes.h"
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
#include <optional>

#define DEBUG_TYPE "tx81-memref-to-llvm"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

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
    // LLVM IR has no f8 types. Load/store the 8-bit pattern as i8 so
    // llvm.load result is a sized LLVM type (builtin f8 is not).
    llvmTypeConverter.addConversion([](Type type) -> std::optional<Type> {
      auto ft = dyn_cast<FloatType>(type);
      if (!ft || ft.getWidth() != 8)
        return std::nullopt;
      return IntegerType::get(type.getContext(), 8);
    });
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
