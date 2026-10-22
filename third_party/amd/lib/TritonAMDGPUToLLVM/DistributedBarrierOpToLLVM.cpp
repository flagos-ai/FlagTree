#include "PatternTritonGPUOpToLLVM.h"
#ifdef __TLE__
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <limits>

// FlagMega-on-Radeon P1-C: RDNA lowering of tle.distributed_barrier for
// group_kind "grid" / "grid_axis" — a sense-reversing global-memory barrier,
// the AMD analogue of the NVVM lowerGridBarrier in
// third_party/tle/dialect/lib/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.cpp.
// The arithmetic (high-bit 0x80000000 phase, master adds 0x80000000-(N-1),
// others add 1) is identical; only the block/grid-id and the atomic/load are
// swapped for AMD primitives: ROCDL workgroup id (targetInfo.programId),
// gpu.grid_dim, and global atomicrmw add with agent scope. "cluster"/"submesh"
// (hardware clusters) are unsupported on RDNA and diagnosed.

using namespace mlir;
using namespace mlir::triton;

namespace {

constexpr int32_t kGridScratchBytes = 4;
constexpr uint32_t kSenseBit = 0x80000000u;

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DistributedBarrierOp> {
  DistributedBarrierOpConversion(LLVMTypeConverter &converter,
                                 const AMD::TargetInfo &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = op->getAttrOfType<StringAttr>("group_kind");
    StringRef k = kind ? kind.getValue() : StringRef("cluster");
    if (k != "grid" && k != "grid_axis")
      return op.emitOpError("distributed_barrier group_kind '")
             << k
             << "' is unsupported on the RDNA backend; only \"grid\" and "
                "\"grid_axis\" (global-memory grid sync) are lowered (RDNA has "
                "no hardware clusters)";
    return lowerGridBarrier(op, rewriter);
  }

  LogicalResult lowerGridBarrier(tle::DistributedBarrierOp op,
                                 ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);

    SmallVector<int32_t> groupShape, groupAxes, domainShape;
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("group_shape"))
      groupShape.assign(a.asArrayRef().begin(), a.asArrayRef().end());
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("group_axes"))
      groupAxes.assign(a.asArrayRef().begin(), a.asArrayRef().end());
    if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape"))
      domainShape.assign(a.asArrayRef().begin(), a.asArrayRef().end());

    const bool isAxisGroup = !domainShape.empty();
    int32_t participantCount = 1;
    SmallVector<int32_t> extentByAxis(domainShape.size(), 1);
    if (isAxisGroup) {
      if (groupShape.size() != groupAxes.size())
        return op.emitOpError("grid axis group shape/axes rank mismatch");
      for (auto [axis, extent] : llvm::zip(groupAxes, groupShape)) {
        if (axis < 0 || axis >= (int32_t)domainShape.size() || extent <= 0 ||
            domainShape[axis] % extent != 0)
          return op.emitOpError("invalid grid axis group descriptor");
        extentByAxis[axis] = extent;
        participantCount *= extent;
      }
    }

    auto scratchOffsetAttr =
        op->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_offset");
    if (!scratchOffsetAttr)
      return op.emitOpError("grid barrier requires global scratch allocation "
                            "before LLVM lowering");
    int32_t scratchOffset = (int32_t)scratchOffsetAttr.getInt();

    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func)
      return op.emitOpError("grid lowering requires LLVM function context");
    int32_t argIdx = (int32_t)func.getNumArguments() + kGlobalScratchBufferOffset;
    if (argIdx < 0 || argIdx >= (int32_t)func.getNumArguments())
      return op.emitOpError("cannot locate global scratch argument");
    Value scratchBase = func.getArgument((unsigned)argIdx);
    auto basePtrTy = dyn_cast<LLVM::LLVMPointerType>(scratchBase.getType());
    if (!basePtrTy)
      return op.emitOpError("global scratch argument must be an LLVM pointer");
    auto i32PtrTy = LLVM::LLVMPointerType::get(ctx, basePtrTy.getAddressSpace());

    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto gridDim = [&](mlir::gpu::Dimension d) -> Value {
      Value g = mlir::gpu::GridDimOp::create(rewriter, loc, d);
      return arith::TruncIOp::create(rewriter, loc, i32Ty, g);
    };
    Value blockIdX = targetInfo.programId(rewriter, loc, mod, ProgramIDDim::X);
    Value blockIdY = targetInfo.programId(rewriter, loc, mod, ProgramIDDim::Y);
    Value blockIdZ = targetInfo.programId(rewriter, loc, mod, ProgramIDDim::Z);
    Value gridDimX = gridDim(mlir::gpu::Dimension::x);
    Value gridDimY = gridDim(mlir::gpu::Dimension::y);
    Value gridDimZ = gridDim(mlir::gpu::Dimension::z);

    Value linearBlockId = b.add(b.mul(blockIdZ, gridDimY), blockIdY);
    linearBlockId = b.add(b.mul(linearBlockId, gridDimX), blockIdX);

    Value groupIndex = b.i32_val(0);
    Value localRank = b.i32_val(0);
    if (isAxisGroup) {
      SmallVector<int32_t> strides(domainShape.size(), 1);
      int32_t stride = 1;
      for (int32_t axis = (int32_t)domainShape.size() - 1; axis >= 0; --axis) {
        strides[axis] = stride;
        stride *= domainShape[axis];
      }
      for (int32_t axis = 0; axis < (int32_t)domainShape.size(); ++axis) {
        Value coord = linearBlockId;
        if (strides[axis] != 1)
          coord = b.udiv(coord, b.i32_val(strides[axis]));
        if (domainShape[axis] != 1)
          coord = b.urem(coord, b.i32_val(domainShape[axis]));
        int32_t extent = extentByAxis[axis];
        int32_t groupsOnAxis = domainShape[axis] / extent;
        Value groupCoord = coord;
        if (extent != 1)
          groupCoord = b.udiv(coord, b.i32_val(extent));
        groupIndex =
            b.add(b.mul(groupIndex, b.i32_val(groupsOnAxis)), groupCoord);
        if (extent != 1) {
          Value localCoord = b.urem(coord, b.i32_val(extent));
          localRank = b.add(b.mul(localRank, b.i32_val(extent)), localCoord);
        }
      }
    }

    Value scratchByteOffset = b.i32_val(scratchOffset);
    if (isAxisGroup)
      scratchByteOffset = b.add(scratchByteOffset,
                                b.mul(groupIndex, b.i32_val(kGridScratchBytes)));
    Value arrivedBytePtr =
        b.gep(basePtrTy, i8Ty, scratchBase, scratchByteOffset);
    Value arrivedPtr = b.bitcast(arrivedBytePtr, i32PtrTy);

    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value isBlock0;
    if (isAxisGroup) {
      isBlock0 = b.icmp_eq(localRank, b.i32_val(0));
    } else {
      isBlock0 = b.and_(b.and_(b.icmp_eq(blockIdX, b.i32_val(0)),
                               b.icmp_eq(blockIdY, b.i32_val(0))),
                        b.icmp_eq(blockIdZ, b.i32_val(0)));
    }

    Value totalCTAs = b.mul(b.mul(gridDimX, gridDimY), gridDimZ);
    Value expectedCTAs =
        isAxisGroup ? b.i32_val(participantCount) : totalCTAs;

    Block *curBlock = rewriter.getInsertionBlock();
    Block *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    Block *workBlock = rewriter.createBlock(endBlock);
    Block *waitBlock = rewriter.createBlock(endBlock);
    waitBlock->addArgument(i32Ty, loc); // old_arrive
    Block *doneBlock = rewriter.createBlock(endBlock);
    Block *workerDoneBlock = rewriter.createBlock(endBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    mlir::gpu::BarrierOp::create(rewriter, loc);
    LLVM::CondBrOp::create(rewriter, loc, isThread0, workBlock, ValueRange{},
                           doneBlock, ValueRange{});

    // Arrive: master CTA adds 0x80000000-(N-1), others add 1, so the sense bit
    // flips exactly once when all expected CTAs have arrived.
    rewriter.setInsertionPointToEnd(workBlock);
    Value expectedMinusOne = b.sub(expectedCTAs, b.i32_val(1));
    Value gpuMasterAdd = b.sub(b.i32_val(kSenseBit), expectedMinusOne);
    Value nb = b.select(isBlock0, gpuMasterAdd, b.i32_val(1));
    Value oldArrive = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::add, arrivedPtr, nb,
        LLVM::AtomicOrdering::release, StringRef("agent"));
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldArrive}, waitBlock);

    // Spin until the sense bit flips, reading with an acquire atomic.
    rewriter.setInsertionPointToEnd(waitBlock);
    Value oldArriveArg = waitBlock->getArgument(0);
    Value curArrive = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::add, arrivedPtr, b.i32_val(0),
        LLVM::AtomicOrdering::acquire, StringRef("agent"));
    Value flipped =
        b.and_(b.xor_(oldArriveArg, curArrive), b.i32_val(kSenseBit));
    Value hasFlipped = b.icmp_ne(flipped, b.i32_val(0));
    LLVM::CondBrOp::create(rewriter, loc, hasFlipped, workerDoneBlock,
                           ValueRange{}, waitBlock, ValueRange{oldArriveArg});

    rewriter.setInsertionPointToEnd(workerDoneBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, endBlock);
    rewriter.setInsertionPointToEnd(doneBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    mlir::gpu::BarrierOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, targetInfo,
                                               benefit);
}

#endif // __TLE__
