#include "Conversion/TleToLLVM.h"
#include "IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"
#include "Conversion/TleToLLVM/ExtractOpToLLVM.h"
#include "Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
#include "Conversion/TleToLLVM/PackOpToLLVM.h"
#include "Conversion/TleToLLVM/ViewOpToLLVM.h"

#ifdef FLAGCX_ENABLED
#include "Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxOpToLLVM.h"
#endif

namespace mlir::triton::iluvatar_tle {

namespace {
struct SmeCopyOpConversion : ConvertOpToLLVMPattern<SmeCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SmeCopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto src = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto dst = unpackLLElements(loc, adaptor.getDst(), rewriter);
    auto offset = unpackLLElements(loc, adaptor.getByteOffset(), rewriter);
    auto active = unpackLLElements(loc, adaptor.getActive(), rewriter);
    if (src.size() != dst.size() ||
        (offset.size() != 1 && offset.size() != src.size()) ||
        (active.size() != 1 && active.size() != src.size()))
      return op.emitOpError("warp-owned operands have incompatible register counts");
    Value tid = rewriter.create<NVVM::ThreadIdXOp>(loc, i32_ty);
    Value leader = b.icmp_eq(b.and_(tid, b.i32_val(63)), b.i32_val(0));
    auto descriptorType = vec_ty(i32_ty, 4);
    for (unsigned i = 0; i < src.size(); ++i) {
      Value predicate = b.and_(leader, active[active.size() == 1 ? 0 : i]);
      auto *current = rewriter.getInsertionBlock();
      auto *continuation = rewriter.splitBlock(current, rewriter.getInsertionPoint());
      auto *issue = rewriter.createBlock(continuation);
      rewriter.setInsertionPointToEnd(current);
      rewriter.create<LLVM::CondBrOp>(loc, predicate, issue, continuation);
      rewriter.setInsertionPointToStart(issue);
      Value address = b.ptrtoint(i64_ty, src[i]);
      Value descriptor = b.undef(descriptorType);
      descriptor = b.insert_element(descriptorType, descriptor,
                                    b.trunc(i32_ty, address), b.i32_val(0));
      descriptor = b.insert_element(descriptorType, descriptor,
                                    b.trunc(i32_ty, b.lshr(address, b.i64_val(32))),
                                    b.i32_val(1));
      descriptor = b.insert_element(descriptorType, descriptor, b.i32_val(-1), b.i32_val(2));
      descriptor = b.insert_element(descriptorType, descriptor, b.i32_val(0), b.i32_val(3));
      LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc,
          op.getNbytes() == 512 ? "llvm.bi.sme.load.1x8b64" : "llvm.bi.sme.load.1x16b64", {},
          {b.ptrtoint(i32_ty, dst[i]), descriptor,
           offset[offset.size() == 1 ? 0 : i], b.i32_val(0)});
      rewriter.create<LLVM::BrOp>(loc, continuation);
      rewriter.setInsertionPointToStart(continuation);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct SmeWaitOpConversion : ConvertOpToLLVMPattern<SmeWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SmeWaitOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    uint64_t flags = 8ull | (static_cast<uint64_t>(op.getMaxPending()) << 23);
    if (op.getDrainReads())
      flags |= 4;
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(), "llvm.bi.sl.waitcnt", {},
                                    {b.i64_val(flags)});
    // Reconverge the elected issuer and consumer lanes without draining
    // unrelated memory queues. This is COREX's __syncwarp intrinsic; a CTA
    // membar would turn a partial SME wait into a full memory-queue drain.
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(), "llvm.bi.sch.barrier",
                                    {}, {});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTleToLLVMPatterns(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo,
                               RewritePatternSet &patterns,
                               PatternBenefit benefit) {
  patterns.add<SmeCopyOpConversion, SmeWaitOpConversion>(typeConverter, benefit);
  mlir::triton::iluvatar_tle::populateExtractTileOpToLLVMPatterns(
      typeConverter, patterns, targetInfo, benefit);
  mlir::triton::iluvatar_tle::populateInsertTileOpToLLVMPatterns(
      typeConverter, patterns, targetInfo, benefit);
  mlir::triton::iluvatar_tle::populateLocalPointersOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::iluvatar_tle::populateRemotePointersOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::iluvatar_tle::populateDistributedBarrierOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  // TLE-Raw: the descriptor plumbing that survives dsl_region inlining.
  // `dsl_region` / `yield` themselves are gone by this point, see
  // TritonIluvatarTleDSLRegionInline.
  mlir::triton::iluvatar_tle::populateExtractOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  mlir::triton::iluvatar_tle::populatePackOpToLLVMPatterns(typeConverter,
                                                           patterns, benefit);
  mlir::triton::iluvatar_tle::populateMemDescAliasOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  // FlagCX ops are lowered to LLVM.
#ifdef FLAGCX_ENABLED
  mlir::triton::iluvatar_tle::populateFlagCxOpToLLVMPatterns(typeConverter,
                                                             patterns, benefit);
#endif
}

} // namespace mlir::triton::iluvatar_tle
