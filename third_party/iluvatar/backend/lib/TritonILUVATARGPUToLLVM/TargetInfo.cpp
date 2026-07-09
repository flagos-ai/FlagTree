#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::ILUVATAR {

namespace {
template <typename T>
LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                     RewriterBase &rewriter, StringRef name,
                                     LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = LLVM::LLVMFuncOp::create(rewriter, loc, name, type,
                                   LLVM::Linkage::External);
  }
  return ret;
}

LLVM::LLVMFuncOp getVprintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("vprintf2");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context), i32_ty};
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context), funcName,
                                  funcType);
}

// Extend integer to int32 and normalize floating-point args to fp32 for CoreX
// vprintf2.
std::pair<Type, Value> printfPromoteValue(RewriterBase &rewriter, Value value,
                                          bool isSigned) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newOp = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    newType = i32_ty;
    if (isSigned) {
      newOp = b.sext(newType, value);
    } else {
      newOp = b.zext(newType, value);
    }
  } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
    newType = f32_ty;
    if (type.isF64())
      newOp = b.fptrunc(newType, value);
    else if (!type.isF32())
      newOp = b.fpext(newType, value);
  }

  return {newType, newOp};
}

LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("__assertfail");
  {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);
  }
  // void __assert_fail(const char * assertion, const char * file, unsigned
  // int line, const char * function);
  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
                             rewriter.getIntegerType(sizeof(size_t) * 8)};
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto funcOp = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx),
                                         funcName, funcType);

  funcOp.setPassthroughAttr(
      ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  return funcOp;
}
} // namespace

int TargetInfo::getWarpSize() const { return 32; }

int TargetInfo::getSharedMemorySize() const {
  // Should return the maximum capacity in kbyte
  return 64 * 1024;
}

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // llvm.bi.vote.ballot do not support i1, so extend it to i32
  cmp = b.zext(i32_ty, cmp);
  auto stringAttr = rewriter.getStringAttr("llvm.bi.vote.ballot");
  SmallVector<Value> operands = {cmp};
  Value asmResult = LLVM::createLLVMIntrinsicCallOp(
                        rewriter, loc, "llvm.bi.vote.ballot", type, operands)
                        ->getResult(0);
  return asmResult;
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  if (isWarpSync) {
    // On Iluvatar MR, lanes in a warp are lockstep-scheduled (__syncwarp is a
    // no-op per the programming guide), so omit warp-level barriers here.
    return;
  } else {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier();
  }
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "cross-CTA shared memory transfers are not supported");
  }
  mlir::LLVM::ILUVATAR::llStore(rewriter, loc, ptr, val, pred);
}

std::optional<TargetInfo::LDSTransLoadParams>
TargetInfo::queryLDSTransLoadParams(int /*bitWidth*/) const {
  return std::nullopt;
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "cross-CTA shared memory transfers are not supported");
  }
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getZeroAttr(elemTy));
  // bool addAliasGroup = localLoadOp && requiresAliasInfoForAsyncOps() &&
  //  isSyncedViaAsyncWait(localLoadOp);
  bool addAliasGroup = localLoadOp && requiresAliasInfoForAsyncOps();
  return mlir::LLVM::ILUVATAR::llLoad(rewriter, loc, ptr, elemTy, pred,
                                      falseVal, {}, triton::CacheModifier::NONE,
                                      addAliasGroup);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::ILUVATAR::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::ILUVATAR::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::ILUVATAR::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::ILUVATAR::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  return LLVM::ILUVATAR::permute(loc, rewriter, a, b, selector);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::ILUVATAR::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__nv_umulhi" : "__nv_umul64hi";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto funcOp = getVprintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value zero = b.i32_val(0);

  Value bufferPtr = b.null(ptr);
  Value bufferSize = b.i32_val(0);

  SmallVector<Value, 16> newArgs;
  if (args.size() >= 1) {
    SmallVector<Type> argTypes;
    for (auto [i, arg] : llvm::enumerate(args)) {
      Type newType;
      Value newArg;
      std::tie(newType, newArg) = printfPromoteValue(
          rewriter, arg, isSigned.empty() ? true : isSigned[i]);
      argTypes.push_back(newType);
      newArgs.push_back(newArg);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
    auto currentPoint = rewriter.saveInsertionPoint();
    auto func =
        rewriter.getInsertionPoint()->getParentOfType<LLVM::LLVMFuncOp>();
    rewriter.setInsertionPointToStart(&func.getBody().front());
    Value one = b.i32_val(1);
    auto allocated =
        LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx, 5), structTy, one,
                               /*alignment=*/0);
    rewriter.restoreInsertionPoint(currentPoint);

    for (const auto &entry : llvm::enumerate(newArgs)) {
      auto index = b.i32_val(entry.index());
      auto fieldPtr = b.gep(ptr_ty(ctx, 5), structTy, allocated,
                            ArrayRef<Value>{zero, index});
      b.store(entry.value(), fieldPtr);
    }
    bufferPtr = b.bitcast(allocated, ptr_ty(ctx, 5));
    bufferPtr = b.addrspacecast(ptr, bufferPtr);

    unsigned argSize = 0;
    for (auto argType : argTypes) {
      if (!isa<LLVM::LLVMPointerType>(argType))
        argSize += argType.getIntOrFloatBitWidth() / 8;
    }
    bufferSize = b.i32_val(argSize);
  }

  SmallVector<Value> operands{formatStrStart, bufferPtr, bufferSize};
  b.call(funcOp, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
  Value fileStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
  Value funcStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
  Value lineNumber = b.i32_val(line);
  Value charSize = b.int_val(sizeof(size_t) * 8, sizeof(char));
  SmallVector<Value> operands = {messageStringVal, fileStringVal, lineNumber,
                                 funcStringVal, charSize};
  b.call(funcOp, operands);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace for now");
  }
  return spaceId;
}

bool TargetInfo::supportVectorizedAtomics() const {
  // Note: not currently tested or used.
  return true;
}

bool TargetInfo::supportsDirectToLDSScattering() const {
  llvm::report_fatal_error("Unsupported architecture for direct to lds loads");
  return false;
}

bool TargetInfo::requiresAliasInfoForAsyncOps() const { return false; }

bool TargetInfo::supportsDirectToLdsLoadBitWidth(int /*bitWidth*/) const {
  return false;
}

} // namespace mlir::triton::ILUVATAR
