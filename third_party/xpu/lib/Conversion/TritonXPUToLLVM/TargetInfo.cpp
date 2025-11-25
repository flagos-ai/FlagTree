//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
// clang-format off
#include "xpu/lib/Conversion/TritonXPUToLLVM/TargetInfo.h"  // TargetInfo

#include "triton/Analysis/UtilityXPU.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// clang-format on

using namespace mlir;

namespace mlir {
namespace triton {
namespace xpu {

template <typename T>
LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     StringRef name,
                                     LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

bool TargetInfo::supportMaximumMinimum() const {
  llvm_unreachable("not impl");
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  llvm_unreachable("not impl");
  return Value();
}

void TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Value val, Value pred) const {
  llvm_unreachable("not impl");
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             const TypeConverter *converter, Value ptr,
                             Type elemTy, Value pred) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleXor(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleUp(ConversionPatternRewriter &rewriter, Location loc,
                            Value val, int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, Value i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::programId(ConversionPatternRewriter &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::XPU::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  llvm_unreachable("not impl");
  return false;
}

bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
    int swizzlingByteWidth) const {
  llvm_unreachable("not impl");
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "_ZN3xpu6umulhiEjj" : "Unsupported";
  return funcName;
}

void TargetInfo::printf(ConversionPatternRewriter &rewriter,
                        Value formatStrStart, int /*formatStrByteCount*/,
                        ValueRange args) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  mlir::Location loc = UnknownLoc::get(ctx);

  LLVM::LLVMFuncOp printFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "printf",
      LLVM::LLVMFunctionType::get(i32_ty, {ptr_ty(ctx)}, true));

  SmallVector<Value, 4> arguments = {formatStrStart};
  SmallVector<Value, 4> argumentsWithoutFmt = {args};
  arguments.append(argumentsWithoutFmt);

  uint32_t envClusterId = -1;
  uint32_t envCoreId = -1;
  std::string envHWIdStr = mlir::triton::tools::getStrEnv("TRITON_PRINT_HW_ID");
  if (!envHWIdStr.empty()) {
    llvm::StringRef envHWIdSRf = envHWIdStr;
    std::pair<llvm::StringRef, llvm::StringRef> envHWIdSRfs =
        envHWIdSRf.split(',');
    llvm::StringRef envClusterIdSRf = envHWIdSRfs.first;
    llvm::StringRef envCoreIdSRf = envHWIdSRfs.second;
    if (envClusterIdSRf.getAsInteger(10, envClusterId) || envClusterId >= 12) {
      llvm::report_fatal_error("Invalid value for TRITON_PRINT_HW_ID: " +
                               envHWIdSRf);
    }
    if (envCoreIdSRf.getAsInteger(10, envCoreId) || envCoreId >= 64) {
      llvm::report_fatal_error("Invalid value for TRITON_PRINT_HW_ID: " +
                               envHWIdSRf);
    }
    auto coreId = rewriter.create<LLVM::XPU::CoreIdOp>(loc, i32_ty);
    auto clusterId = rewriter.create<LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(ctx), i32_val(0));
    auto condCoreId = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, envCoreId);
    auto condClusterId =
        rewriter.create<LLVM::ConstantOp>(loc, i32_ty, envCoreId);
    auto cmpCoreId = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                   coreId, condCoreId);
    auto cmpClusterId = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, clusterId, condClusterId);
    auto cond = rewriter.create<LLVM::AndOp>(loc, cmpCoreId, cmpClusterId);

    auto thenBlock = rewriter.splitBlock(rewriter.getInsertionBlock(),
                                         rewriter.getInsertionPoint());
    auto mergeBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

    rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
    rewriter.create<LLVM::CondBrOp>(loc, cond, thenBlock, mergeBlock);

    rewriter.setInsertionPointToStart(thenBlock);
    call(printFn, arguments).getResult();
    rewriter.create<LLVM::BrOp>(loc, mergeBlock);

    rewriter.setInsertionPointToStart(mergeBlock);
  } else {
    call(printFn, arguments).getResult();
  }
}

void TargetInfo::assertFail(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  LLVM::LLVMFuncOp assertFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__assert_fail",
      LLVM::LLVMFunctionType::get(
          void_ty(ctx), {ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx)}, true));
  while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
    loc = callLoc.getCallee();

  if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
    file = fileLineColLoc.getFilename();
    line = fileLineColLoc.getLine();
  }
  assertFn.setPassthroughAttr(
      ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  Value messageString =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", message);
  Value fileString =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", file);
  Value funcString =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", func);
  Value lineNumber = i32_val(line);
  SmallVector<Value> operands = {messageString, fileString, lineNumber,
                                 funcString};
  call(assertFn, operands);
}

uint32_t TargetInfo::getXPUArch() const { return this->xpu_arch; }
uint32_t TargetInfo::getXPUBufferSize() const { return this->buffer_size; }
bool TargetInfo::getXPUIsUseMaskZero() const { return this->isUseMaskZero; }

} // namespace xpu
} // namespace triton
} // namespace mlir
