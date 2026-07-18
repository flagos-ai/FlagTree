#include "tle/dialect/include/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>

namespace {

using namespace mlir;
namespace tle = mlir::triton::tle;

constexpr llvm::StringLiteral kGroupKindAttr = "group_kind";
constexpr llvm::StringLiteral kGroupShapeAttr = "group_shape";
constexpr llvm::StringLiteral kGroupMaskAttr = "group_mask";
constexpr llvm::StringLiteral kTTGSharedAttr = "ttg.shared";
constexpr llvm::StringLiteral kTTGGlobalScratchSizeAttr =
    "ttg.global_scratch_memory_size";
constexpr llvm::StringLiteral kTTGGlobalScratchAlignAttr =
    "ttg.global_scratch_memory_alignment";
constexpr llvm::StringLiteral kSubmeshScratchOffsetAttr =
    "tle.submesh_barrier_scratch_offset";
constexpr llvm::StringLiteral kGridScratchOffsetAttr =
    "tle.grid_barrier_scratch_offset";
constexpr int32_t kSubmeshScratchAlignment = 16;
constexpr int32_t kSubmeshScratchBytes = 8;
constexpr int32_t kSubmeshCounterOffsetBytes = 0;
constexpr int32_t kSubmeshPhaseOffsetBytes = 4;
constexpr int32_t kGridScratchAlignment = 4;
constexpr int32_t kGridScratchBytes = 4;
constexpr int32_t kGridArrivedOffsetBytes = 0;

FailureOr<int32_t> getOrCreateSubmeshScratchOffset(ModuleOp mod) {
  if (auto existing =
          mod->getAttrOfType<IntegerAttr>(kSubmeshScratchOffsetAttr)) {
    int64_t value = existing.getInt();
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
      return failure();
    return static_cast<int32_t>(value);
  }

  auto sharedAttr = mod->getAttrOfType<IntegerAttr>(kTTGSharedAttr);
  if (!sharedAttr)
    return failure();

  int64_t currentShared = sharedAttr.getInt();
  if (currentShared < 0)
    return failure();

  int64_t offset =
      llvm::alignTo(currentShared, int64_t{kSubmeshScratchAlignment});
  int64_t newShared = offset + kSubmeshScratchBytes;
  if (newShared > std::numeric_limits<int32_t>::max())
    return failure();

  auto i32Ty = IntegerType::get(mod.getContext(), 32);
  mod->setAttr(kTTGSharedAttr, IntegerAttr::get(i32Ty, newShared));
  mod->setAttr(kSubmeshScratchOffsetAttr, IntegerAttr::get(i32Ty, offset));
  return static_cast<int32_t>(offset);
}

FailureOr<int32_t> getOrCreateGridScratchOffset(ModuleOp mod) {
  if (auto existing = mod->getAttrOfType<IntegerAttr>(kGridScratchOffsetAttr)) {
    int64_t value = existing.getInt();
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
      return failure();
    return static_cast<int32_t>(value);
  }

  auto *ctx = mod.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  int64_t currentSize = 0;
  if (auto sizeAttr =
          mod->getAttrOfType<IntegerAttr>(kTTGGlobalScratchSizeAttr)) {
    currentSize = sizeAttr.getInt();
    if (currentSize < 0)
      return failure();
  } else {
    mod->setAttr(kTTGGlobalScratchSizeAttr, IntegerAttr::get(i32Ty, 0));
  }

  int64_t currentAlign = 1;
  if (auto alignAttr =
          mod->getAttrOfType<IntegerAttr>(kTTGGlobalScratchAlignAttr)) {
    currentAlign = alignAttr.getInt();
    if (currentAlign <= 0)
      return failure();
  } else {
    mod->setAttr(kTTGGlobalScratchAlignAttr, IntegerAttr::get(i32Ty, 1));
  }

  int64_t offset = llvm::alignTo(currentSize, int64_t{kGridScratchAlignment});
  int64_t newSize = offset + kGridScratchBytes;
  if (newSize > std::numeric_limits<int32_t>::max())
    return failure();
  int64_t newAlign = std::max(currentAlign, int64_t{kGridScratchAlignment});

  mod->setAttr(kTTGGlobalScratchSizeAttr, IntegerAttr::get(i32Ty, newSize));
  mod->setAttr(kTTGGlobalScratchAlignAttr, IntegerAttr::get(i32Ty, newAlign));
  mod->setAttr(kGridScratchOffsetAttr, IntegerAttr::get(i32Ty, offset));
  return static_cast<int32_t>(offset);
}

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DistributedBarrierOp> {
  using ConvertOpToLLVMPattern<
      tle::DistributedBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult lowerClusterBarrier(tle::DistributedBarrierOp op,
                                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = rewriter.getContext();
    auto unit = UnitAttr::get(ctx);
    // Cluster arrive/wait does not provide CTA-wide synchronization semantics
    // for local shared-memory hazards. Add CTA barriers around it so
    // distributed_barrier behaves as a full barrier for each participating CTA.
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.create<NVVM::ClusterArriveOp>(op.getLoc(), unit);
    rewriter.create<NVVM::ClusterWaitOp>(op.getLoc(), unit);
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerGridBarrier(tle::DistributedBarrierOp op,
                                 ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);

    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return op.emitOpError("cannot find parent module for grid lowering");

    auto scratchOffsetOr = getOrCreateGridScratchOffset(mod);
    if (failed(scratchOffsetOr)) {
      return op.emitOpError(
          "failed to reserve global scratch for grid barrier");
    }
    int32_t scratchOffset = *scratchOffsetOr;

    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      return op.emitOpError("grid lowering requires LLVM function context");
    }
    int32_t argIdx = static_cast<int32_t>(func.getNumArguments()) +
                     kGlobalScratchBufferOffset;
    if (argIdx < 0 || argIdx >= static_cast<int32_t>(func.getNumArguments())) {
      return op.emitOpError(
          "cannot locate global scratch argument for grid barrier lowering");
    }
    Value globalScratchBase = func.getArgument(static_cast<unsigned>(argIdx));
    auto globalPtrTy =
        dyn_cast<LLVM::LLVMPointerType>(globalScratchBase.getType());
    if (!globalPtrTy) {
      return op.emitOpError("global scratch argument must be an LLVM pointer");
    }

    auto globalI32PtrTy =
        LLVM::LLVMPointerType::get(ctx, globalPtrTy.getAddressSpace());
    Value arrivedBytePtr =
        b.gep(globalPtrTy, i8Ty, globalScratchBase,
              b.i32_val(scratchOffset + kGridArrivedOffsetBytes));
    Value arrivedPtr = b.bitcast(arrivedBytePtr, globalI32PtrTy);

    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value blockIdX = rewriter.create<NVVM::BlockIdXOp>(loc, i32Ty);
    Value blockIdY = rewriter.create<NVVM::BlockIdYOp>(loc, i32Ty);
    Value blockIdZ = rewriter.create<NVVM::BlockIdZOp>(loc, i32Ty);
    Value isBlock0X = b.icmp_eq(blockIdX, b.i32_val(0));
    Value isBlock0Y = b.icmp_eq(blockIdY, b.i32_val(0));
    Value isBlock0Z = b.icmp_eq(blockIdZ, b.i32_val(0));
    Value isBlock0 = b.and_(b.and_(isBlock0X, isBlock0Y), isBlock0Z);
    Value workerPred = isThread0;

    Value gridDimX = rewriter.create<NVVM::GridDimXOp>(loc, i32Ty);
    Value gridDimY = rewriter.create<NVVM::GridDimYOp>(loc, i32Ty);
    Value gridDimZ = rewriter.create<NVVM::GridDimZOp>(loc, i32Ty);
    Value totalCTAs = b.mul(gridDimX, gridDimY);
    totalCTAs = b.mul(totalCTAs, gridDimZ);

    Block *curBlock = rewriter.getInsertionBlock();
    Block *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    Block *workBlock = rewriter.createBlock(endBlock);
    Block *waitBlock = rewriter.createBlock(endBlock);
    waitBlock->addArgument(i32Ty, loc); // old_arrive
    Block *doneBlock = rewriter.createBlock(endBlock);
    Block *workerDoneBlock = rewriter.createBlock(endBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.create<LLVM::CondBrOp>(loc, workerPred, workBlock, ValueRange{},
                                    doneBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(workBlock);
    Value expectedMinusOne = b.sub(totalCTAs, b.i32_val(1));
    Value gpuMasterAdd = b.sub(b.i32_val(0x80000000u), expectedMinusOne);
    Value nb = b.select(isBlock0, gpuMasterAdd, b.i32_val(1));

    auto emitAtomAddReleaseGpu = [&](Value ptr, Value addVal) -> Value {
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &atom = *ptxBuilder.create<>("atom.add.release.gpu.u32");
      auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
      auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "l");
      auto *addOpr = ptxBuilder.newOperand(addVal, "r");
      atom(dstOpr, ptrOpr, addOpr);
      return ptxBuilder.launch(rewriter, loc, i32Ty);
    };
    auto emitLoadAcquireGpu = [&](Value ptr) -> Value {
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &ld = *ptxBuilder.create<>("ld.acquire.gpu.u32");
      auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
      auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "l");
      ld(dstOpr, ptrOpr);
      return ptxBuilder.launch(rewriter, loc, i32Ty);
    };

    Value oldArrive = emitAtomAddReleaseGpu(arrivedPtr, nb);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{oldArrive}, waitBlock);

    rewriter.setInsertionPointToEnd(waitBlock);
    Value oldArriveArg = waitBlock->getArgument(0);
    Value currentArrive = emitLoadAcquireGpu(arrivedPtr);
    Value xorVal = b.xor_(oldArriveArg, currentArrive);
    Value flippedBit = b.and_(xorVal, b.i32_val(0x80000000u));
    Value hasFlipped = b.icmp_ne(flippedBit, b.i32_val(0));
    rewriter.create<LLVM::CondBrOp>(loc, hasFlipped, workerDoneBlock,
                                    ValueRange{}, waitBlock,
                                    ValueRange{oldArriveArg});

    rewriter.setInsertionPointToEnd(workerDoneBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToEnd(doneBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerSubmeshBarrier(tle::DistributedBarrierOp op,
                                    ConversionPatternRewriter &rewriter) const {
    auto maskAttr = op->getAttrOfType<DenseI32ArrayAttr>(kGroupMaskAttr);
    if (!maskAttr) {
      return op.emitOpError("submesh lowering requires static group_mask attr");
    }

    SmallVector<int32_t> subgroupMask(maskAttr.asArrayRef().begin(),
                                      maskAttr.asArrayRef().end());
    if (subgroupMask.empty()) {
      return op.emitOpError("submesh lowering requires non-empty group_mask");
    }
    if (llvm::any_of(subgroupMask, [](int32_t v) { return v < 0; })) {
      return op.emitOpError(
          "submesh lowering requires non-negative group_mask entries");
    }

    if (auto shapeAttr =
            op->getAttrOfType<DenseI32ArrayAttr>(kGroupShapeAttr)) {
      int64_t subgroupFromShape = 1;
      for (int32_t dim : shapeAttr.asArrayRef())
        subgroupFromShape *= dim;
      if (subgroupFromShape != static_cast<int64_t>(subgroupMask.size())) {
        return op.emitOpError() << "group_shape product (" << subgroupFromShape
                                << ") must match group_mask size ("
                                << subgroupMask.size() << ")";
      }
    }

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);

    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return op.emitOpError("cannot find parent module for submesh lowering");

    auto scratchOffsetOr = getOrCreateSubmeshScratchOffset(mod);
    if (failed(scratchOffsetOr)) {
      return op.emitOpError(
          "failed to reserve shared memory scratch for submesh barrier");
    }
    int32_t scratchOffset = *scratchOffsetOr;

    auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!globalSmem) {
      return op.emitOpError("global_smem symbol is missing; submesh barrier "
                            "lowering requires shared memory base");
    }

    auto sharedPtrTy = LLVM::LLVMPointerType::get(
        ctx, static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
    auto clusterPtrTy = LLVM::LLVMPointerType::get(
        ctx, static_cast<unsigned>(NVVM::NVVMMemorySpace::SharedCluster));

    Value sharedBase = rewriter.create<LLVM::AddressOfOp>(loc, globalSmem);
    sharedBase = b.bitcast(sharedBase, sharedPtrTy);

    Value counterLocalPtr =
        b.gep(sharedPtrTy, i8Ty, sharedBase,
              b.i32_val(scratchOffset + kSubmeshCounterOffsetBytes));
    Value phaseLocalPtr =
        b.gep(sharedPtrTy, i8Ty, sharedBase,
              b.i32_val(scratchOffset + kSubmeshPhaseOffsetBytes));

    int32_t leaderCTAId = subgroupMask.front();
    Value leaderCTA = b.i32_val(leaderCTAId);
    Value counterPtr = rewriter.create<NVVM::MapaOp>(
        loc, clusterPtrTy, counterLocalPtr, leaderCTA);
    Value phasePtr = rewriter.create<NVVM::MapaOp>(loc, clusterPtrTy,
                                                   phaseLocalPtr, leaderCTA);

    Value clusterCTAId = rewriter.create<triton::nvgpu::ClusterCTAIdOp>(loc);
    Value isParticipant = b.false_val();
    for (int32_t member : subgroupMask) {
      Value isMember = b.icmp_eq(clusterCTAId, b.i32_val(member));
      isParticipant = b.or_(isParticipant, isMember);
    }

    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value workerPred = b.and_(isParticipant, isThread0);
    Value isLeaderCTA = b.icmp_eq(clusterCTAId, leaderCTA);
    Value doInit = b.and_(isLeaderCTA, isThread0);

    auto unit = UnitAttr::get(ctx);
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *postInitBlock = entryBlock->splitBlock(rewriter.getInsertionPoint());
    Block *initBlock = rewriter.createBlock(postInitBlock);

    rewriter.setInsertionPointToEnd(entryBlock);
    // Conservative initialization fence: ensure all CTAs observe counter reset
    // before any subgroup arrivals in this barrier instance.
    rewriter.create<NVVM::ClusterArriveOp>(loc, unit);
    rewriter.create<NVVM::ClusterWaitOp>(loc, unit);
    rewriter.create<LLVM::CondBrOp>(loc, doInit, initBlock, ValueRange{},
                                    postInitBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<LLVM::StoreOp>(loc, b.i32_val(0), counterPtr);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, postInitBlock);

    rewriter.setInsertionPointToStart(postInitBlock);
    rewriter.create<NVVM::ClusterArriveOp>(loc, unit);
    rewriter.create<NVVM::ClusterWaitOp>(loc, unit);

    Block *curBlock = rewriter.getInsertionBlock();
    Block *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    Block *workBlock = rewriter.createBlock(endBlock);
    Block *waitBlock = rewriter.createBlock(endBlock);
    waitBlock->addArgument(i32Ty, loc);
    Block *lastBlock = rewriter.createBlock(endBlock);
    Block *doneBlock = rewriter.createBlock(endBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.create<LLVM::CondBrOp>(loc, workerPred, workBlock, ValueRange{},
                                    doneBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(workBlock);
    Value oldPhase =
        rewriter
            .create<LLVM::AtomicRMWOp>(
                loc, LLVM::AtomicBinOp::add, phasePtr, b.i32_val(0),
                LLVM::AtomicOrdering::acquire, StringRef("device"))
            .getResult();
    Value prevCount =
        rewriter
            .create<LLVM::AtomicRMWOp>(
                loc, LLVM::AtomicBinOp::add, counterPtr, b.i32_val(1),
                LLVM::AtomicOrdering::acq_rel, StringRef("device"))
            .getResult();
    Value arrived = b.add(prevCount, b.i32_val(1));
    Value isLast = b.icmp_eq(arrived, b.i32_val(subgroupMask.size()));
    rewriter.create<LLVM::CondBrOp>(loc, isLast, lastBlock, ValueRange{},
                                    waitBlock, ValueRange{oldPhase});

    rewriter.setInsertionPointToEnd(waitBlock);
    Value expectedPhase = waitBlock->getArgument(0);
    Value currentPhase =
        rewriter
            .create<LLVM::AtomicRMWOp>(
                loc, LLVM::AtomicBinOp::add, phasePtr, b.i32_val(0),
                LLVM::AtomicOrdering::acquire, StringRef("device"))
            .getResult();
    Value keepWaiting = b.icmp_eq(currentPhase, expectedPhase);
    rewriter.create<LLVM::CondBrOp>(loc, keepWaiting, waitBlock,
                                    ValueRange{expectedPhase}, doneBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(lastBlock);
    rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, counterPtr,
        b.i32_val(-static_cast<int64_t>(subgroupMask.size())),
        LLVM::AtomicOrdering::release, StringRef("device"));
    rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, phasePtr, b.i32_val(1),
        LLVM::AtomicOrdering::release, StringRef("device"));
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, doneBlock);

    rewriter.setInsertionPointToEnd(doneBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kindAttr = op->getAttrOfType<StringAttr>(kGroupKindAttr)) {
      if (kindAttr.getValue() == "grid")
        return lowerGridBarrier(op, rewriter);
      if (kindAttr.getValue() == "submesh")
        return lowerSubmeshBarrier(op, rewriter);
      return lowerClusterBarrier(op, rewriter);
    }
    return lowerClusterBarrier(op, rewriter);
  }
};

} // namespace

void tle::populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit);
}
