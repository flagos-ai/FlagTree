#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

constexpr int kBarrierCountBitWidth = 29;
constexpr int kBarrierPhaseMask = ((1ULL << (32 - kBarrierCountBitWidth)) - 1);
constexpr int kInitCountPos = 32;

namespace {

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *ldsBarrierInitBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));
    rewriter.setInsertionPointToEnd(curBlock);
    auto id = getThreadId(rewriter, loc);
    auto pred = b.icmp_eq(id, b.i32_val(0));
    LLVM::CondBrOp::create(rewriter, loc, pred, ldsBarrierInitBlock, endBlock);
    rewriter.setInsertionPointToEnd(ldsBarrierInitBlock);
    // Phase changes when underflow is detected (pending count becomes
    // negative). The provided count from the user assumes that phase changes
    // when pending count reaches zero, so make the adjustment here.
    Value count = b.i64_val(op.getCount() - 1);
    Value val = b.or_(b.shl(count, b.i64_val(kInitCountPos)), count);
    b.store(val, smemObj.getBase());
    LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);
    rewriter.setInsertionPointToStart(endBlock);
    // Synchronize the whole CTA, so all waves see the LDS barrier
    b.barrier();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ArriveBarrierOp> {
  ArriveBarrierOpConversion(LLVMTypeConverter &converter,
                            const AMD::TargetInfo &targetInfo,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto count = adaptor.getCount();
    Value base = smemObj.getBase();

    if (targetInfo.getISAFamily() == AMD::ISAFamily::GFX1250) {
      // Hardware named-barrier arrive. The intrinsic
      // `ds.atomic.barrier.arrive.rtn.b64` only exists on gfx12.5+ (verified via
      // `llc -mcpu=`: every other AMD family, RDNA1-4 and CDNA1-4, reports
      // "Cannot select"), so it is gated here and emulated below otherwise.
      // NOTE: The LLVM intrisic expects an i64_ty for count (update value)
      // But count cannot be more than 32bits according to ISA docs.
      Value priorState =
          LLVM::createLLVMIntrinsicCallOp(
              rewriter, loc, "llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64",
              i64_ty, {base, b.i64_val(count)})
              .getResult(0);
      Value priorPhase = b.and_(
          i32_ty, b.i32_val(kBarrierPhaseMask),
          b.trunc(i32_ty,
                  b.lshr(priorState, b.i64_val(kBarrierCountBitWidth))));
      rewriter.replaceOp(op, priorPhase);
      return success();
    }

    // Software emulation of the barrier-arrive for targets without the hardware
    // intrinsic. A cmpxchg loop on the i64 LDS barrier word reproduces the
    // decrement / phase-flip / reload semantics while keeping the bit layout
    // identical to Init/WaitBarrier so the three interoperate:
    //   bits [0 : kBarrierCountBitWidth-1] = pending arrival count
    //   bits [kBarrierCountBitWidth : 31]  = phase (kBarrierPhaseMask wide)
    //   bits [kInitCountPos : 63]          = reload count (== init count - 1)
    // InitBarrier stores `count-1` in both halves, so the phase flips exactly
    // when the count-th arrival underflows the pending field, then the pending
    // field reloads from the high half for the next phase.
    const uint64_t pendingMask = (1ULL << kBarrierCountBitWidth) - 1;
    const uint64_t highMask = 0xFFFFFFFF00000000ULL;
    Value cnt64 = b.i64_val(count);

    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    endBlock->addArgument(i32_ty, loc); // prior phase from the winning CAS
    auto *loopBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));
    loopBlock->addArgument(i64_ty, loc); // current expected barrier word

    // Seed the loop with a plain read of the barrier word.
    rewriter.setInsertionPointToEnd(curBlock);
    Value seed = b.load(i64_ty, base);
    LLVM::BrOp::create(rewriter, loc, ValueRange{seed}, loopBlock);

    rewriter.setInsertionPointToStart(loopBlock);
    Value prior = loopBlock->getArgument(0);
    Value pending = b.and_(prior, b.i64_val(pendingMask));
    Value reload =
        b.and_(b.lshr(prior, b.i64_val(kInitCountPos)), b.i64_val(pendingMask));
    Value phase = b.and_(b.lshr(prior, b.i64_val(kBarrierCountBitWidth)),
                         b.i64_val((uint64_t)kBarrierPhaseMask));
    // Underflow (phase flip) when the pending field cannot absorb this arrival.
    Value isUnderflow = b.icmp_ult(pending, cnt64);
    Value subP = b.sub(pending, cnt64);
    // On underflow reload the pending pool: reload + 1 + (pending - count).
    Value underP = b.add(b.add(reload, b.i64_val(1)), subP);
    Value newPending =
        b.and_(b.select(isUnderflow, underP, subP), b.i64_val(pendingMask));
    Value phaseDec = b.and_(b.sub(phase, b.i64_val(1)),
                            b.i64_val((uint64_t)kBarrierPhaseMask));
    Value newPhase = b.select(isUnderflow, phaseDec, phase);
    Value newLow =
        b.or_(newPending, b.shl(newPhase, b.i64_val(kBarrierCountBitWidth)));
    Value newState = b.or_(b.and_(prior, b.i64_val(highMask)), newLow);

    auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, base, prior, newState, LLVM::AtomicOrdering::acq_rel,
        LLVM::AtomicOrdering::monotonic, StringRef("workgroup"));
    Value loaded = b.extract_val(i64_ty, cmpxchg, 0);
    Value casSuccess = b.extract_val(i1_ty, cmpxchg, 1);
    Value priorPhase = b.trunc(i32_ty, phase);
    LLVM::CondBrOp::create(rewriter, loc, casSuccess, endBlock,
                           ValueRange{priorPhase}, loopBlock,
                           ValueRange{loaded});

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.replaceOp(op, endBlock->getArgument(0));
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::WaitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    Value phase = adaptor.getPhase();
    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *waitBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));
    rewriter.setInsertionPointToEnd(curBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange(), waitBlock);
    rewriter.setInsertionPointToStart(waitBlock);
    // Sleep for the minimum number of clocks. 64*SIMM16[6:0] = 64 * 1 = 64
    // clocks.
    ROCDL::SSleepOp::create(rewriter, loc, 1);
    Value curState = b.load(i64_ty, smemObj.getBase());
    Value curPhase = b.and_(
        i32_ty, b.i32_val(kBarrierPhaseMask),
        b.trunc(i32_ty, b.lshr(curState, b.i64_val(kBarrierCountBitWidth))));
    Value phaseChanged = b.icmp_ne(curPhase, phase);
    LLVM::CondBrOp::create(rewriter, loc, phaseChanged, endBlock, waitBlock);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<InitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, targetInfo, benefit);
}
