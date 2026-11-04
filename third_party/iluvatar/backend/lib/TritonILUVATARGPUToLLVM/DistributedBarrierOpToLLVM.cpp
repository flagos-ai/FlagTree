// Lower the backend-neutral grid barrier to COREX global atomics.
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::tle {
namespace {
using namespace mlir;
using namespace mlir::triton;
constexpr int32_t kScratchBytes = 4;

struct GridBarrierLowering : public ConvertOpToLLVMPattern<DistributedBarrierOp> {
  using ConvertOpToLLVMPattern<DistributedBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(DistributedBarrierOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kind = op->getAttrOfType<StringAttr>("group_kind");
    if (!kind)
      return rewriter.notifyMatchFailure(op, "distributed barrier group kind is missing");
    if (kind.getValue() == "block") {
      rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
      rewriter.eraseOp(op);
      return success();
    }
    if (kind.getValue() != "grid" && kind.getValue() != "grid_axis")
      return rewriter.notifyMatchFailure(op, "COREX supports block/grid barriers only");
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i8 = IntegerType::get(ctx, 8);
    auto i32 = IntegerType::get(ctx, 32);
    auto mod = op->getParentOfType<ModuleOp>();
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!mod || !func)
      return op.emitOpError("grid barrier requires module/function");
    auto offset = op->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_offset");
    if (!offset)
      return op.emitOpError("grid barrier scratch offset is missing");
    const unsigned scratchArg = func.getNumArguments() + kGlobalScratchBufferOffset;
    if (scratchArg >= func.getNumArguments())
      return op.emitOpError("global scratch argument is missing");
    TritonLLVMOpBuilder b(loc, rewriter);
    Value base = func.getArgument(scratchArg);
    auto ptrTy = cast<LLVM::LLVMPointerType>(base.getType());
    auto i32PtrTy = LLVM::LLVMPointerType::get(ctx, ptrTy.getAddressSpace());
    // A grid-axis barrier has one counter per disjoint group.  The builder
    // records static group shape/axes and the launch domain; derive both the
    // local rank (which CTA participates) and the group counter index here.
    SmallVector<int32_t> groupShape, groupAxes, domainShape;
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("group_shape"))
      groupShape.assign(a.asArrayRef().begin(), a.asArrayRef().end());
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("group_axes"))
      groupAxes.assign(a.asArrayRef().begin(), a.asArrayRef().end());
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape"))
      domainShape.assign(a.asArrayRef().begin(), a.asArrayRef().end());
    const bool axisGroup = kind.getValue() == "grid_axis";
    if (axisGroup && (domainShape.empty() || groupShape.size() != groupAxes.size()))
      return op.emitOpError("grid axis group metadata is incomplete");
    SmallVector<int32_t> extents(domainShape.size(), 1);
    if (axisGroup) {
      for (auto [axis, extent] : llvm::zip(groupAxes, groupShape)) {
        if (axis < 0 || axis >= static_cast<int32_t>(domainShape.size()) ||
            extent <= 0 || domainShape[axis] % extent != 0)
          return op.emitOpError("invalid grid axis group descriptor");
        extents[axis] = extent;
      }
    }
    Value bx = rewriter.create<NVVM::BlockIdXOp>(loc, i32);
    Value by = rewriter.create<NVVM::BlockIdYOp>(loc, i32);
    Value bz = rewriter.create<NVVM::BlockIdZOp>(loc, i32);
    Value dx = rewriter.create<NVVM::GridDimXOp>(loc, i32);
    Value dy = rewriter.create<NVVM::GridDimYOp>(loc, i32);
    Value dz = rewriter.create<NVVM::GridDimZOp>(loc, i32);
    Value linear = b.add(b.add(b.mul(bz, b.mul(dx, dy)), b.mul(by, dx)), bx);
    Value groupIndex = b.i32_val(0);
    Value localRank = b.i32_val(0);
    int32_t participantCount = 1;
    if (axisGroup) {
      SmallVector<int32_t> strides(domainShape.size(), 1);
      int32_t stride = 1;
      for (int axis = static_cast<int>(domainShape.size()) - 1; axis >= 0; --axis) {
        strides[axis] = stride;
        stride *= domainShape[axis];
      }
      for (int axis = 0; axis < static_cast<int>(domainShape.size()); ++axis) {
        Value coord = linear;
        if (strides[axis] != 1)
          coord = b.udiv(coord, b.i32_val(strides[axis]));
        if (domainShape[axis] != 1)
          coord = b.urem(coord, b.i32_val(domainShape[axis]));
        int32_t extent = extents[axis];
        int32_t groupsOnAxis = domainShape[axis] / extent;
        groupIndex = b.add(b.mul(groupIndex, b.i32_val(groupsOnAxis)),
                           b.udiv(coord, b.i32_val(extent)));
        localRank = b.add(b.mul(localRank, b.i32_val(extent)),
                          b.urem(coord, b.i32_val(extent)));
        participantCount *= extent;
      }
    }
    Value groupOffset = axisGroup
                            ? b.mul(groupIndex, b.i32_val(kScratchBytes))
                            : b.i32_val(0);
    Value ptrOffset = b.add(b.i32_val(offset.getInt()), groupOffset);
    Value ptr = b.gep(ptrTy, i8, base, ptrOffset);
    ptr = b.bitcast(ptr, i32PtrTy);

    Value total = axisGroup ? b.i32_val(participantCount)
                            : b.mul(b.mul(dx, dy), dz);
    Value tid = getThreadId(rewriter, loc);
    Value worker = b.icmp_eq(tid, b.i32_val(0));
    Value leader = axisGroup ? b.icmp_eq(localRank, b.i32_val(0))
                             : b.icmp_eq(linear, b.i32_val(0));


    Block *cur = rewriter.getInsertionBlock();
    Block *end = cur->splitBlock(rewriter.getInsertionPoint());
    Block *work = rewriter.createBlock(end);
    Block *wait = rewriter.createBlock(end);
    rewriter.setInsertionPointToEnd(cur);
    // Every producing warp must finish publishing its global-memory writes
    // before the elected thread announces this CTA's arrival. A block barrier
    // alone does not drain those writes to other SMs on COREX.
    rewriter.create<NVVM::MembarOp>(loc, NVVM::MemScopeKind::GPU);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    // Every CTA contributes one arrival. CTA 0 uses the high-bit master
    // increment to release the generation once all other CTAs have arrived.
    // The CFG mirrors the working hand-written CUDA barrier: worker and
    // non-worker paths converge DIRECTLY on the single exit block (one
    // predecessor set, no intermediate join blocks); extra merge blocks get
    // mis-reconverged by the BI-V150 tmsk machinery on large kernels.
    rewriter.create<LLVM::CondBrOp>(loc, worker, work, ValueRange{}, end, ValueRange{});

    rewriter.setInsertionPointToEnd(work);
    Value expectedMinusOne = b.sub(total, b.i32_val(1));
    Value masterIncrement =
        b.sub(b.i32_val(static_cast<int32_t>(0x80000000u)), expectedMinusOne);
    Value increment = b.select(leader, masterIncrement, b.i32_val(1));
    // COREX lowers LLVM atomics on global pointers to the native LSA atomic
    // instructions.  PTX inline assembly is NVIDIA-specific and is rejected
    // by the BI-V150 assembler, so keep the barrier in backend-neutral LLVM.
    // The all-thread fence above publishes payloads. Release/acquire atomics
    // separately order the counter protocol; acquire invalidation must remain
    // for consumers on COREX's non-coherent per-SM caches.
    Value old = rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, ptr, increment,
        LLVM::AtomicOrdering::release, /*syncscope=*/StringRef(),
        /*alignment=*/4);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, wait);

    rewriter.setInsertionPointToEnd(wait);
    // `old` dominates this block and is loop-invariant; do NOT thread it
    // through the block argument. An argument-free self-loop keeps the spin
    // in the same simple shape clang emits for hand-written CUDA barriers.
    //
    Value previous = old;
    Value current = rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::_or, ptr, b.i32_val(0),
        LLVM::AtomicOrdering::acquire, /*syncscope=*/StringRef(),
        /*alignment=*/4);
    Value flipped = b.and_(b.xor_(previous, current),
                           b.i32_val(static_cast<int32_t>(0x80000000u)));
    Value complete = b.icmp_ne(flipped, b.i32_val(0));
    rewriter.create<LLVM::CondBrOp>(loc, complete, end, ValueRange{}, wait, ValueRange{});

    rewriter.setInsertionPointToStart(end);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GridBarrierLowering>(typeConverter, benefit);
}
} // namespace mlir::triton::tle
