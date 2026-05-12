#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::triton::MUSA;

namespace {

Location getCurrentLoc(RewriterBase &rewriter) {
  if (Operation *parent = rewriter.getBlock()->getParentOp())
    return parent->getLoc();
  return UnknownLoc::get(rewriter.getContext());
}

LLVM::LLVMFuncOp getPrintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  constexpr StringRef funcName("printf");
  if (Operation *funcOp = moduleOp.lookupSymbol(funcName))
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();
  auto funcType =
      LLVM::LLVMFunctionType::get(i32_ty, {ptr_ty(context)}, /*isVarArg=*/true);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context), funcName,
                                  funcType);
}

std::pair<Type, Value> printfPromoteValue(RewriterBase &rewriter, Value value,
                                          bool isSigned) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newValue = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    newType = i32_ty;
    newValue = isSigned ? b.sext(newType, value).getResult()
                        : b.zext(newType, value).getResult();
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    newType = f64_ty;
    newValue = b.fpext(newType, value).getResult();
  }

  return {newType, newValue};
}

} // namespace

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  Value pred = cmp;
  if (!cmp.getType().isInteger(1)) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (cmp.getType().isIntOrIndex()) {
      pred =
          b.icmp_ne(cmp, b.int_val(cmp.getType().getIntOrFloatBitWidth(), 0));
    } else {
      Value zero = LLVM::ConstantOp::create(
          rewriter, loc, cmp.getType(), rewriter.getZeroAttr(cmp.getType()));
      pred = LLVM::FCmpOp::create(rewriter, loc, rewriter.getI1Type(),
                                  LLVM::FCmpPredicate::one, cmp, zero);
    }
  }
  // MTGPU backend lowers the sync vote intrinsics; emitting the legacy
  // non-sync ballot leaves an unsupported intrinsic in the DAG legalization
  // path and can wedge llc in release builds.
  return LLVM::createLLVMIntrinsicCallOp(
             rewriter, loc, "llvm.musa.vote.ballot.sync", type, {pred})
      .getResult(0);
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  if (targets == triton::gpu::AddrSpace::Local) {
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.syncthreads.lm",
                                    TypeRange{}, {});
  }
  if (targets != triton::gpu::AddrSpace::Local) {
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.barrier0",
                                    TypeRange{}, {});
  }
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.syncwarp",
                                  TypeRange{}, {});
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  assert(!ctaId.has_value() && "cross-CTA shared stores are not supported");
  LLVM::MUSA::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation * /*localLoadOp*/) const {
  assert(!ctaId.has_value() && "cross-CTA shared loads are not supported");
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getZeroAttr(elemTy));
  return LLVM::MUSA::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::MUSA::shuffleXor(loc, rewriter, val, i, 32);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::MUSA::shuffleUp(loc, rewriter, val, i, 32);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::MUSA::shuffleIdx(loc, rewriter, val, i, 32);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::MUSA::shuffleIdx(loc, rewriter, val, i, 32);
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  return LLVM::MUSA::permute(loc, rewriter, a, b, selector);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::MUSA::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(RewriterBase & /*rewriter*/, Location /*loc*/,
                            SmallVector<Value> & /*acc*/,
                            triton::ReduceOp /*op*/,
                            unsigned /*numLaneToReduce*/,
                            unsigned /*interleave*/) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  (void)resultElementTy;
  return "";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto funcOp = getPrintfDeclaration(rewriter);
  SmallVector<Value> operands{formatStrStart};
  for (auto [i, arg] : llvm::enumerate(args)) {
    Type newType;
    Value newArg;
    std::tie(newType, newArg) = printfPromoteValue(
        rewriter, arg, isSigned.empty() ? true : isSigned[i]);
    (void)newType;
    operands.push_back(newArg);
  }
  LLVM::CallOp::create(rewriter, getCurrentLoc(rewriter), funcOp, operands);
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
  llvm::SmallString<96> assertFormat(
      "[MUSA_KERNEL_ASSERT] %s:%u: %s: Assertion `%s' failed.\n");
  assertFormat.push_back('\0');
  Value formatValue =
      LLVM::addStringToModule(loc, rewriter, "assertFormat_", assertFormat);

  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');

  Value messageValue =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
  Value fileValue =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
  Value funcValue =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
  Value lineValue = b.i32_val(line);

  printf(rewriter, formatValue, assertFormat.size_in_bytes(),
         {fileValue, lineValue, funcValue, messageValue},
         {false, false, false, false});
  barrier(loc, rewriter, triton::gpu::AddrSpace::All);
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.exit", TypeRange{},
                                  {});
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace))
    return 3;
  return 0;
}

bool TargetInfo::supportVectorizedAtomics() const { return false; }
