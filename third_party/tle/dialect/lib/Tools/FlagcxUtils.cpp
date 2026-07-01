#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::tle {
using namespace mlir;

static const llvm::StringMap<StringRef> runtimeNames = {
    {"getLocalPeFunction", "flagcxDevCommGetIntraRank"},
    {"getNumPesFunction", "flagcxDevCommGetIntraSize"},
    {"getReadSignalFunction", "flagcxDevNetReadSignal"}};

static inline LLVM::LLVMFuncOp createFuncInstance(const char *funcName,
                                                  ModuleOp module,
                                                  ArrayRef<Type> argTypes,
                                                  Type returnType) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;
  auto funcType = LLVM::LLVMFunctionType::get(returnType, argTypes, false);

  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);

  func.setLinkage(LLVM::Linkage::External);
  return func;
}

static inline Value getFlagcxMemPtr(mlir::Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    Value memPtrInt) {
  auto ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
  return rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, memPtrInt);
}

LLVM::CallOp getNumPesFunCall(mlir::Location loc,
                              ConversionPatternRewriter &rewriter,
                              Value memPtrInt) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto func = createFuncInstance(
      runtimeNames.lookup("getNumPesFunction").data(), module, {PtrTy}, i32Ty);

  auto comm_dev_ptr = getFlagcxMemPtr(loc, rewriter, memPtrInt);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr});
}

LLVM::CallOp getReadSignalFunCall(mlir::Location loc,
                                  ConversionPatternRewriter &rewriter,
                                  Value memPtrInt) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto func = createFuncInstance(
      runtimeNames.lookup("getNumPesFunction").data(), module, {PtrTy}, i32Ty);

  auto comm_dev_ptr = getFlagcxMemPtr(loc, rewriter, memPtrInt);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr});
}

LLVM::CallOp getLocalPeFuncCall(mlir::Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value memPtrInt) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto func = createFuncInstance(
      runtimeNames.lookup("getLocalPeFunction").data(), module, {PtrTy}, i32Ty);

  auto comm_dev_ptr = getFlagcxMemPtr(loc, rewriter, memPtrInt);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr});
}

} // namespace mlir::triton::tle
