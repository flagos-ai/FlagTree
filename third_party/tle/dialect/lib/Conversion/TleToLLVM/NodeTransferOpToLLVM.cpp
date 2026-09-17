/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/NodeTransferOpToLLVM.h"

#include <optional>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

enum class NodeTransferKind { Put, Get };

static LLVM::LLVMFuncOp getOrInsertNetFromComm(ModuleOp module,
                                               MLIRContext *ctx) {
  const char *funcName = "flagcxDevNetGetFromCommS";
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;

  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto funcTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, i32Ty}, false);
  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcTy);
  func.setLinkage(LLVM::Linkage::External);
  return func;
}

static LLVM::LLVMFuncOp getOrInsertNetPut(ModuleOp module, MLIRContext *ctx) {
  const char *funcName = "flagcxDevNetPutS";
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  SmallVector<Type> argTypes{ptrTy, ptrTy, i32Ty, i32Ty, ptrTy,
                             i64Ty, ptrTy, i64Ty, i64Ty, i32Ty};
  auto funcTy = LLVM::LLVMFunctionType::get(voidTy, argTypes, false);
  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcTy);
  func.setLinkage(LLVM::Linkage::External);
  return func;
}

static LLVM::LLVMFuncOp getOrInsertNetGet(ModuleOp module, MLIRContext *ctx) {
  const char *funcName = "flagcxDevNetGetS";
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  SmallVector<Type> argTypes{ptrTy, ptrTy, i32Ty, i32Ty, ptrTy,
                             i64Ty, ptrTy, i64Ty, i64Ty, i32Ty};
  auto funcTy = LLVM::LLVMFunctionType::get(voidTy, argTypes, false);
  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcTy);
  func.setLinkage(LLVM::Linkage::External);
  return func;
}

static std::optional<int64_t> getConstantInt(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

static LogicalResult
lowerNodeTransfer(Location loc, Operation *op, Value srcHandle, Value dstHandle,
                  Value commHandle, Value peer, Value srcOffset,
                  Value dstOffset, Value nelems, Value netIdx,
                  int64_t elemBytes, int64_t coopKindValue,
                  NodeTransferKind kind, std::optional<int64_t> constantNelems,
                  ConversionPatternRewriter &rewriter) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    return rewriter.notifyMatchFailure(op, "expected a parent module");

  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  Value dstMem = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, dstHandle);
  Value srcMem = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, srcHandle);
  Value comm = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, commHandle);

  Value srcByteOffset = srcOffset;
  Value dstByteOffset = dstOffset;
  Value byteCount = nelems;
  if (elemBytes != 1) {
    Value elemBytesValue =
        rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    srcByteOffset =
        rewriter.create<arith::MulIOp>(loc, srcOffset, elemBytesValue);
    dstByteOffset =
        rewriter.create<arith::MulIOp>(loc, dstOffset, elemBytesValue);
    byteCount = rewriter.create<arith::MulIOp>(loc, nelems, elemBytesValue);
  }

  auto emitTransfer = [&]() {
    LLVM::LLVMFuncOp getNet = getOrInsertNetFromComm(module, ctx);
    auto getNetCall = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{ptrTy}, FlatSymbolRefAttr::get(getNet),
        ValueRange{comm, netIdx});
    Value teamKind = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(2));
    Value coopKind = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(coopKindValue));

    if (kind == NodeTransferKind::Put) {
      LLVM::LLVMFuncOp put = getOrInsertNetPut(module, ctx);
      rewriter.create<LLVM::CallOp>(
          loc, TypeRange{}, FlatSymbolRefAttr::get(put),
          ValueRange{getNetCall.getResult(), comm, teamKind, peer, dstMem,
                     dstByteOffset, srcMem, srcByteOffset, byteCount,
                     coopKind});
      return;
    }

    LLVM::LLVMFuncOp get = getOrInsertNetGet(module, ctx);
    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, FlatSymbolRefAttr::get(get),
        ValueRange{getNetCall.getResult(), comm, teamKind, peer, srcMem,
                   srcByteOffset, dstMem, dstByteOffset, byteCount, coopKind});
  };

  if (!constantNelems) {
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    Value hasElements = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::sgt, nelems, zero);
    rewriter.create<triton::AssertOp>(
        loc, hasElements,
        rewriter.getStringAttr(
            "node remote transfer requires nelems > 0; use an unmasked scalar "
            "transfer, or with offsets = tl.arange(0, N), require 1 <= valid_n "
            "<= N"));
  }

  emitTransfer();
  return success();
}

struct NodePutOpConversion : public ConvertOpToLLVMPattern<tle::NodePutOp> {
  NodePutOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::NodePutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(lowerNodeTransfer(
            op.getLoc(), op.getOperation(), adaptor.getSrc(),
            adaptor.getDstMem(), adaptor.getComm(), adaptor.getPeer(),
            adaptor.getSrcOffset(), adaptor.getDstOffset(), adaptor.getNelems(),
            adaptor.getNetIdx(), op.getElemBytesAttr().getInt(),
            static_cast<int32_t>(op.getCoopKind()), NodeTransferKind::Put,
            getConstantInt(op.getNelems()), rewriter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct NodeGetOpConversion : public ConvertOpToLLVMPattern<tle::NodeGetOp> {
  NodeGetOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::NodeGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(lowerNodeTransfer(
            op.getLoc(), op.getOperation(), adaptor.getSrc(),
            adaptor.getDstMem(), adaptor.getComm(), adaptor.getPeer(),
            adaptor.getSrcOffset(), adaptor.getDstOffset(), adaptor.getNelems(),
            adaptor.getNetIdx(), op.getElemBytesAttr().getInt(),
            static_cast<int32_t>(op.getCoopKind()), NodeTransferKind::Get,
            getConstantInt(op.getNelems()), rewriter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::tle::populateNodeTransferOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<NodePutOpConversion, NodeGetOpConversion>(typeConverter,
                                                         benefit);
}
