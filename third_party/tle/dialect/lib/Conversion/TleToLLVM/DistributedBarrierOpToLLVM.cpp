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
constexpr llvm::StringLiteral kGroupAxesAttr = "group_axes";
constexpr llvm::StringLiteral kGroupMaskAttr = "group_mask";
constexpr llvm::StringLiteral kGroupDomainShapeAttr = "group_domain_shape";
constexpr llvm::StringLiteral kTTGSharedAttr = "ttg.shared";
constexpr llvm::StringLiteral kTTGGlobalScratchSizeAttr =
    "ttg.global_scratch_memory_size";
constexpr llvm::StringLiteral kTTGGlobalScratchAlignAttr =
    "ttg.global_scratch_memory_alignment";
constexpr llvm::StringLiteral kSubmeshScratchOffsetAttr =
    "tle.submesh_barrier_scratch_offset";
constexpr llvm::StringLiteral kGlobalScratchMemoryOffsetAttr =
    "ttg.global_scratch_memory_offset";
constexpr int32_t kSubmeshScratchAlignment = 16;
constexpr int32_t kSubmeshScratchBytes = 8;
constexpr int32_t kSubmeshCounterOffsetBytes = 0;
constexpr int32_t kSubmeshPhaseOffsetBytes = 4;
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

    auto scratchOffsetAttr =
        op->getAttrOfType<IntegerAttr>(kGlobalScratchMemoryOffsetAttr);
    if (!scratchOffsetAttr) {
      return op.emitOpError()
             << "grid lowering requires " << kGlobalScratchMemoryOffsetAttr
             << "; run global scratch allocation before TLE LLVM lowering";
    }

    int64_t scratchOffsetValue = scratchOffsetAttr.getInt();
    if (scratchOffsetValue < 0 ||
        scratchOffsetValue > std::numeric_limits<int32_t>::max()) {
      return op.emitOpError("grid scratch offset is out of i32 range");
    }
    int32_t scratchOffset = static_cast<int32_t>(scratchOffsetValue);

    auto kindAttr = op->getAttrOfType<StringAttr>(kGroupKindAttr);
    const bool isAxisGroup =
        kindAttr && kindAttr.getValue() == "grid_axis_group";
    SmallVector<int32_t> axisGroupDomainShape;
    SmallVector<int32_t> axisGroupAxes;
    SmallVector<int32_t> axisGroupShape;
    if (isAxisGroup) {
      auto domainShapeAttr =
          op->getAttrOfType<DenseI32ArrayAttr>(kGroupDomainShapeAttr);
      auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>(kGroupAxesAttr);
      auto shapeAttr = op->getAttrOfType<DenseI32ArrayAttr>(kGroupShapeAttr);
      if (!domainShapeAttr || !axesAttr || !shapeAttr ||
          axesAttr.asArrayRef().empty()) {
        return op.emitOpError(
            "grid_axis_group lowering requires domain shape, group axes, and group shape");
      }
      axisGroupDomainShape.assign(domainShapeAttr.asArrayRef().begin(),
                                  domainShapeAttr.asArrayRef().end());
      axisGroupAxes.assign(axesAttr.asArrayRef().begin(),
                           axesAttr.asArrayRef().end());
      axisGroupShape.assign(shapeAttr.asArrayRef().begin(),
                            shapeAttr.asArrayRef().end());
    }

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

    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value blockIdX = rewriter.create<NVVM::BlockIdXOp>(loc, i32Ty);
    Value blockIdY = rewriter.create<NVVM::BlockIdYOp>(loc, i32Ty);
    Value blockIdZ = rewriter.create<NVVM::BlockIdZOp>(loc, i32Ty);
    Value gridDimX = rewriter.create<NVVM::GridDimXOp>(loc, i32Ty);
    Value gridDimY = rewriter.create<NVVM::GridDimYOp>(loc, i32Ty);
    Value gridDimZ = rewriter.create<NVVM::GridDimZOp>(loc, i32Ty);
    Value linearBlockId = b.add(blockIdY, b.mul(gridDimY, blockIdZ));
    linearBlockId = b.add(blockIdX, b.mul(gridDimX, linearBlockId));
    Value totalCTAs = b.mul(gridDimX, gridDimY);
    totalCTAs = b.mul(totalCTAs, gridDimZ);

    Value isLeader = b.icmp_eq(linearBlockId, b.i32_val(0));
    Value participantCount = totalCTAs;
    Value counterIndex = b.i32_val(0);
    if (isAxisGroup) {
      int32_t domainSize = 1;
      SmallVector<uint8_t> isGroupAxis(axisGroupDomainShape.size(), 0);
      SmallVector<int32_t> groupDimByAxis(axisGroupDomainShape.size(), 1);
      for (auto [axis, groupDim] :
           llvm::zip_equal(axisGroupAxes, axisGroupShape)) {
        isGroupAxis[static_cast<size_t>(axis)] = 1;
        groupDimByAxis[static_cast<size_t>(axis)] = groupDim;
      }
      for (int32_t dim : axisGroupDomainShape)
        domainSize *= dim;

      Value remaining = b.urem(linearBlockId, b.i32_val(domainSize));
      SmallVector<Value> coordinates(axisGroupDomainShape.size());
      for (int32_t axis =
               static_cast<int32_t>(axisGroupDomainShape.size()) - 1;
           axis >= 0; --axis) {
        Value dim =
            b.i32_val(axisGroupDomainShape[static_cast<size_t>(axis)]);
        coordinates[static_cast<size_t>(axis)] = b.urem(remaining, dim);
        remaining = b.udiv(remaining, dim);
      }

      Value rankInGroup = b.i32_val(0);
      int32_t axisGroupSize = 1;
      for (size_t axis = 0; axis < axisGroupDomainShape.size(); ++axis) {
        int32_t dim = axisGroupDomainShape[axis];
        if (isGroupAxis[axis]) {
          int32_t groupDim = groupDimByAxis[axis];
          Value subgroupRank = b.urem(coordinates[axis], b.i32_val(groupDim));
          rankInGroup = b.add(b.mul(rankInGroup, b.i32_val(groupDim)),
                              subgroupRank);
          int32_t groupCountOnAxis = dim / groupDim;
          if (groupCountOnAxis > 1) {
            Value subgroupIndex =
                b.udiv(coordinates[axis], b.i32_val(groupDim));
            counterIndex =
                b.add(b.mul(counterIndex, b.i32_val(groupCountOnAxis)),
                      subgroupIndex);
          }
          axisGroupSize *= groupDim;
        } else {
          counterIndex =
              b.add(b.mul(counterIndex, b.i32_val(dim)), coordinates[axis]);
        }
      }
      isLeader = b.icmp_eq(rankInGroup, b.i32_val(0));
      participantCount = b.i32_val(axisGroupSize);
    }
    Value workerPred = isThread0;

    auto globalI32PtrTy =
        LLVM::LLVMPointerType::get(ctx, globalPtrTy.getAddressSpace());
    Value counterByteOffset =
        b.add(b.i32_val(scratchOffset + kGridArrivedOffsetBytes),
              b.mul(counterIndex, b.i32_val(4)));
    Value arrivedBytePtr =
        b.gep(globalPtrTy, i8Ty, globalScratchBase, counterByteOffset);
    Value arrivedPtr = b.bitcast(arrivedBytePtr, globalI32PtrTy);

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
    Value expectedMinusOne = b.sub(participantCount, b.i32_val(1));
    Value gpuMasterAdd = b.sub(b.i32_val(0x80000000u), expectedMinusOne);
    Value nb = b.select(isLeader, gpuMasterAdd, b.i32_val(1));

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
    auto emitPollBackoff = [&]() {
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &sleep = *ptxBuilder.create("nanosleep.u32 20");
      sleep();
      ptxBuilder.launch(rewriter, loc, void_ty(ctx));
    };
    Value oldArrive = emitAtomAddReleaseGpu(arrivedPtr, nb);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{oldArrive}, waitBlock);

    rewriter.setInsertionPointToEnd(waitBlock);
    Value oldArriveArg = waitBlock->getArgument(0);
    emitPollBackoff();
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
      if (kindAttr.getValue() == "grid" ||
          kindAttr.getValue() == "grid_axis_group")
        return lowerGridBarrier(op, rewriter);
      if (kindAttr.getValue() == "submesh")
        return lowerSubmeshBarrier(op, rewriter);
      return lowerClusterBarrier(op, rewriter);
    }
    return lowerClusterBarrier(op, rewriter);
  }
};

struct ClusterCTAIdOpConversion
    : public ConvertOpToLLVMPattern<tle::ClusterCTAIdOp> {
  using ConvertOpToLLVMPattern<tle::ClusterCTAIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::ClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::BlockInClusterIdXOp>(
        op, rewriter.getI32Type());
    return success();
  }
};

} // namespace

void tle::populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion, ClusterCTAIdOpConversion>(
      typeConverter, benefit);
}
