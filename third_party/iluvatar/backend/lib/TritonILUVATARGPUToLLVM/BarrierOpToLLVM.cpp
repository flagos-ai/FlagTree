#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#ifdef __ILUVATAR_TLE__
#include "IR/Dialect.h"
#endif

using namespace mlir;
using namespace mlir::triton;

#ifdef __ILUVATAR_TLE__

namespace {

namespace ilu_tle = mlir::triton::iluvatar_tle;

static Value getBarrierPtr(ConversionPatternRewriter &rewriter, Location loc,
                           Value allocStruct, Type elemTy,
                           const LLVMTypeConverter *typeConverter) {
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
      loc, allocStruct, typeConverter->convertType(elemTy), rewriter);
  return smemObj.getBase();
}

static Value getBarrierI32Ptr(ConversionPatternRewriter &rewriter, Location loc,
                              Value i64Ptr) {
  // Opaque pointers: same ptr value is used for i32 atomics on the low half.
  (void)rewriter;
  (void)loc;
  return i64Ptr;
}

static Value getArriveCountPtr(ConversionPatternRewriter &rewriter,
                               Location loc, Value counterI32Ptr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type ptrTy = ptr_ty(rewriter.getContext(), /*addressSpace=*/3);
  return b.gep(ptrTy, i32_ty, counterI32Ptr, ArrayRef<LLVM::GEPArg>{1});
}

static void lowerInitBarrier(ConversionPatternRewriter &rewriter, Location loc,
                             Operation *op, Value ptr, uint32_t count) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value tid = getThreadId(rewriter, loc);
  Value isThread0 = b.icmp_eq(tid, b.i32_val(0));

  Block *cur = rewriter.getInsertionBlock();
  Block *end = rewriter.splitBlock(cur, rewriter.getInsertionPoint());
  Block *init = rewriter.createBlock(end);

  rewriter.setInsertionPointToEnd(cur);
  LLVM::CondBrOp::create(rewriter, loc, isThread0, init, end);

  rewriter.setInsertionPointToEnd(init);
  Value counterPtr = getBarrierI32Ptr(rewriter, loc, ptr);
  Value arriveCountPtr = getArriveCountPtr(rewriter, loc, counterPtr);
  LLVM::StoreOp::create(rewriter, loc, b.i32_val(0), counterPtr,
                        /*alignment=*/4);
  LLVM::StoreOp::create(rewriter, loc, b.i32_val(static_cast<int32_t>(count)),
                        arriveCountPtr, /*alignment=*/4);
  LLVM::BrOp::create(rewriter, loc, end);

  rewriter.setInsertionPointToStart(end);
  rewriter.eraseOp(op);
}

static void lowerWaitBarrier(ConversionPatternRewriter &rewriter, Location loc,
                             Operation *op, Value ptr, Value expectPhase,
                             Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value counterPtr = getBarrierI32Ptr(rewriter, loc, ptr);
  Value arriveCountPtr = getArriveCountPtr(rewriter, loc, counterPtr);
  StringRef scope = "workgroup";

  Block *cur = rewriter.getInsertionBlock();
  Block *end = rewriter.splitBlock(cur, rewriter.getInsertionPoint());
  Block *spin = rewriter.createBlock(end);

  rewriter.setInsertionPointToEnd(cur);
  if (pred) {
    Block *doWait = rewriter.createBlock(spin);
    LLVM::CondBrOp::create(rewriter, loc, pred, doWait, end);
    rewriter.setInsertionPointToEnd(doWait);
    LLVM::BrOp::create(rewriter, loc, spin);
  } else {
    LLVM::BrOp::create(rewriter, loc, spin);
  }

  rewriter.setInsertionPointToEnd(spin);
  // i32 atomicrmw (supported); mirror WS named-barrier "atomic load".
  Value counter = LLVM::AtomicRMWOp::create(
                      rewriter, loc, LLVM::AtomicBinOp::add, counterPtr,
                      b.i32_val(0), LLVM::AtomicOrdering::acquire, scope,
                      /*alignment=*/4)
                      .getResult();
  Value arriveCount =
      LLVM::LoadOp::create(rewriter, loc, i32_ty, arriveCountPtr,
                           /*alignment=*/4);
  Value safeCount =
      b.select(b.icmp_eq(arriveCount, b.i32_val(0)), b.i32_val(1), arriveCount);
  Value phaseParity = b.and_(b.udiv(counter, safeCount), b.i32_val(1));
  // Spin while parity == expect (phase not yet complete); matches PTX
  // test_wait.parity with empty-barrier phase XOR applied by the pipe pass.
  Value done = b.icmp_ne(phaseParity, expectPhase);
  LLVM::CondBrOp::create(rewriter, loc, done, end, spin);

  rewriter.setInsertionPointToStart(end);
  rewriter.eraseOp(op);
}

static void lowerArriveBarrier(ConversionPatternRewriter &rewriter,
                               Location loc, Operation *op, Value ptr,
                               int32_t count) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value counterPtr = getBarrierI32Ptr(rewriter, loc, ptr);
  StringRef scope = "workgroup";

  int32_t partitionThreads = count > 0 ? count : 1;
  Value tid = getThreadId(rewriter, loc);
  Value isElect =
      b.icmp_eq(b.urem(tid, b.i32_val(partitionThreads)), b.i32_val(0));

  Block *cur = rewriter.getInsertionBlock();
  Block *end = rewriter.splitBlock(cur, rewriter.getInsertionPoint());
  Block *arrive = rewriter.createBlock(end);

  rewriter.setInsertionPointToEnd(cur);
  LLVM::CondBrOp::create(rewriter, loc, isElect, arrive, end);

  rewriter.setInsertionPointToEnd(arrive);
  int32_t arriveDelta = count;
  if (arriveDelta <= 0)
    arriveDelta = 1;
  LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add, counterPtr,
                            b.i32_val(arriveDelta),
                            LLVM::AtomicOrdering::acq_rel, scope,
                            /*alignment=*/4);
  LLVM::BrOp::create(rewriter, loc, end);

  rewriter.setInsertionPointToStart(end);
  rewriter.eraseOp(op);
}

// Iluvatar TLE software mbarrier ops, emulated with shared-memory atomics.
struct TleInitBarrierOpConversion
    : public ConvertOpToLLVMPattern<ilu_tle::InitBarrierOp> {
  using ConvertOpToLLVMPattern<ilu_tle::InitBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ilu_tle::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value ptr = getBarrierPtr(rewriter, loc, adaptor.getAlloc(),
                              op.getAlloc().getType().getElementType(),
                              getTypeConverter());
    lowerInitBarrier(rewriter, loc, op, ptr, op.getCount());
    return success();
  }
};

struct TleWaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<ilu_tle::WaitBarrierOp> {
  using ConvertOpToLLVMPattern<ilu_tle::WaitBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ilu_tle::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value ptr = getBarrierPtr(rewriter, loc, adaptor.getAlloc(),
                              op.getAlloc().getType().getElementType(),
                              getTypeConverter());
    lowerWaitBarrier(rewriter, loc, op, ptr, adaptor.getPhase(), Value());
    return success();
  }
};

struct TleArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<ilu_tle::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern<
      ilu_tle::ArriveBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ilu_tle::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value ptr = getBarrierPtr(rewriter, loc, adaptor.getAlloc(),
                              op.getAlloc().getType().getElementType(),
                              getTypeConverter());
    lowerArriveBarrier(rewriter, loc, op, ptr, op.getCount());
    return success();
  }
};

} // namespace

#endif // __ILUVATAR_TLE__

void mlir::triton::ILUVATAR::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo) {
  if (targetInfo.getArch() != "ivcore11")
    return;
#ifdef __ILUVATAR_TLE__
  patterns.add<TleInitBarrierOpConversion, TleWaitBarrierOpConversion,
               TleArriveBarrierOpConversion>(typeConverter, benefit);
#else
  (void)typeConverter;
  (void)patterns;
  (void)benefit;
#endif
}
