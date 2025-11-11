#include "triton/Analysis/Membar.h"

#include "python/src/plugin.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using getLoadIncNum_RankedTensorTypeFunc = unsigned (*)(
    RankedTensorType &, int, int, llvm::ArrayRef<unsigned>, unsigned);

using getLoadIncNum_MemDescTypeFunc = unsigned (*)(MemDescType &, int, int,
                                                   llvm::ArrayRef<unsigned>,
                                                   unsigned);

DEFINE_LOAD_FUNC(getLoadIncNum_RankedTensorType)
DEFINE_LOAD_FUNC(getLoadIncNum_MemDescType)

namespace mlir {

// type: 0 all | 1 del W from other R |2 del R from other W
void BlockInfo::erase(BlockInfo &other, int type) {
  if (type == 0) {
    for (auto &sri : other.syncReadIntervals)
      syncReadIntervals.erase(sri);
    for (auto &swi : other.syncWriteIntervals)
      syncWriteIntervals.erase(swi);
  } else if (type == 1) {
    for (auto &sri : other.syncReadIntervals)
      syncWriteIntervals.erase(sri);
  } else if (type == 2) {
    for (auto &swi : other.syncWriteIntervals)
      syncReadIntervals.erase(swi);
  }
}

// for debug
void BlockInfo::printIntervals() {
  if (syncReadIntervals.size() > 0 || syncWriteIntervals.size() > 0) {
    std::cout << " syncReadIntervals";
    for (auto &lhs : syncReadIntervals)
      std::cout << " [" << lhs.start() << ", " << lhs.end() << "] ";
    std::cout << "" << std::endl;
    std::cout << " syncWriteIntervals";
    for (auto &lhs : syncWriteIntervals)
      std::cout << " [" << lhs.start() << ", " << lhs.end() << "] ";
    std::cout << "" << std::endl;
  }
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<gpu::BarrierOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo.syncWriteIntervals.insert(
                    allocation->getAllocatedInterval(bufferId));
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo.syncReadIntervals.insert(
                    allocation->getAllocatedInterval(bufferId));
            }
          }
        }
      }
    }
    // XXX(Keren): This is a hack as we cannot set side effects for dot ops, but
    // on hopper they do have side effects. Need to clean it up
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      for (auto value : dotOp.getOperands()) {
        for (auto bufferId : allocation->getBufferIds(value)) {
          if (bufferId != Allocation::InvalidBufferId)
            curBlockInfo.syncReadIntervals.insert(
                allocation->getAllocatedInterval(bufferId));
        }
      }
    }
    // Scratch buffer is considered as both shared memory write & read
    auto bufferId = allocation->getBufferId(op);
    if (bufferId != Allocation::InvalidBufferId) {
      curBlockInfo.syncWriteIntervals.insert(
          allocation->getAllocatedInterval(bufferId));
      curBlockInfo.syncReadIntervals.insert(
          allocation->getAllocatedInterval(bufferId));
    }
  }

  bool useBar = true;
  int num_stages = triton::gpu::TritonGPUDialect::getDotNumStages(
      op->getParentOfType<ModuleOp>());
  if (num_stages > 1 && isa<triton::gpu::LocalLoadOp>(op)) {
    auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(op);
    Value src = localLoadOp.getSrc();
    Value dst = localLoadOp.getResult();
    auto srcTy = src.getType().cast<MemDescType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    OpBuilder::InsertionGuard g(*builder);
    auto cnt_ty = builder->getIntegerType(64);
    auto insertBuffId = [](Value val, BlockInfo &tmpInfo,
                           Allocation *allocation) {
      for (auto bufferId : allocation->getBufferIds(val)) {
        if (bufferId != Allocation::InvalidBufferId) {
          tmpInfo.syncWriteIntervals.insert(
              allocation->getAllocatedInterval(bufferId));
        }
      }
    };
    // a use inser but b use load, this should add alu.bar
    if (srcLayout.isa<mlir::triton::gpu::BlockedEncodingAttr>() &&
        dstLayout.isa<mlir::triton::gpu::SharedEncodingAttr>() &&
        blockInfo->isIntersected(curBlockInfo)) {
      builder->setInsertionPoint(op);
      builder->create<LLVM::CallIntrinsicOp>(
          op->getLoc(), void_ty(builder->getContext()),
          "llvm.bi.sl.barrier.alu", ValueRange({}));
      useBar = false;
      blockInfo->syncReadIntervals.clear();
    } else if (srcLayout.isa<mlir::triton::gpu::SharedEncodingAttr>() &&
               dstLayout.isa<mlir::triton::gpu::DotOperandEncodingAttr>() &&
               blockInfo->isIntersected(curBlockInfo)) {
      builder->setInsertionPoint(op);
      auto cnt_ty = builder->getIntegerType(64);
      Value wait_cnt = nullptr;
      BlockInfo tmpInfo;
      auto capability =
          getNVIDIAComputeCapability(op->getParentOfType<ModuleOp>());
      if (capability < 80 || num_stages == 2) {
        wait_cnt = builder->create<LLVM::ConstantOp>(
            op->getLoc(), cnt_ty, IntegerAttr::get(cnt_ty, 12));
        blockInfo->sync();
      } else {
        int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(
            op->getParentOfType<ModuleOp>());
        auto srcSharedLayout =
            dyn_cast<mlir::triton::gpu::SharedEncodingAttr>(srcLayout);
        auto dotOperandLayout =
            dstLayout.cast<mlir::triton::gpu::DotOperandEncodingAttr>();
        if (dotOperandLayout.getUseSme() > 0) {
          Operation *tmp = nullptr;
          unsigned asyncOrSmeNum = 0;
          for (Operation *dop : dst.getUsers()) {
            if (auto transOp = llvm::dyn_cast<triton::TransOp>(dop)) {
              tmp = transOp.getSrc().getDefiningOp();
            } else {
              tmp = dop;
            }
            if (auto dotOp = llvm::dyn_cast<mlir::triton::DotOp>(tmp)) {
              // check A & B use sme|async
              if (dotOperandLayout.getOpIdx() == 0) {
                insertBuffId(src, tmpInfo, allocation);
                DEFINE_CALL_LOAD_FUNC(iluvatar, getLoadIncNum_RankedTensorType)
                asyncOrSmeNum = func(dstTy, capability, numWarps,
                                     srcSharedLayout.getOrder(), 0);
                Value dotB = dotOp.getOperand(1);
                Operation *bOp = dotB.getDefiningOp();
                if (bOp) {
                  if (auto convertBOp =
                          dyn_cast<triton::gpu::LocalLoadOp>(bOp)) {
                    Value dstB = convertBOp.getResult();
                    auto dotBLayout =
                        dstB.getType()
                            .cast<RankedTensorType>()
                            .getEncoding()
                            .cast<mlir::triton::gpu::DotOperandEncodingAttr>();
                    if (dotBLayout.getUseSme() > 0) {
                      insertBuffId(convertBOp.getSrc(), tmpInfo, allocation);
                      Value srcB = convertBOp.getSrc();
                      auto srcBTy = srcB.getType().cast<MemDescType>();
                      auto sharedBLayout =
                          srcBTy.getEncoding()
                              .cast<mlir::triton::gpu::SharedEncodingAttr>();
                      DEFINE_CALL_LOAD_FUNC(iluvatar, getLoadIncNum_MemDescType)
                      unsigned incBNum = func(srcBTy, capability, numWarps,
                                              sharedBLayout.getOrder(), 1);
                      asyncOrSmeNum += incBNum;
                    }
                  }
                }
              } else if (dotOperandLayout.getOpIdx() == 1) {
                DEFINE_CALL_LOAD_FUNC(iluvatar, getLoadIncNum_RankedTensorType)
                asyncOrSmeNum = func(dstTy, capability, numWarps,
                                     srcSharedLayout.getOrder(), 1);
                insertBuffId(src, tmpInfo, allocation);
              }
              break;
            }
          }
          unsigned long long int waitCnt = 8;
          int exponent = 23;
          while (asyncOrSmeNum > 0) {
            int tail = asyncOrSmeNum % 2;
            if (tail > 0)
              waitCnt += (1 << exponent);
            asyncOrSmeNum = asyncOrSmeNum >> 1;
            exponent++;
          }
          wait_cnt = builder->create<LLVM::ConstantOp>(
              op->getLoc(), cnt_ty,
              IntegerAttr::get(cnt_ty, waitCnt)); // only smem intrinsic
        } else {
          wait_cnt = builder->create<LLVM::ConstantOp>(
              op->getLoc(), cnt_ty, IntegerAttr::get(cnt_ty, 4));
          insertBuffId(src, tmpInfo, allocation);
        }
        blockInfo->erase(tmpInfo);
      }
      useBar = false;
      builder->create<LLVM::CallIntrinsicOp>(
          op->getLoc(), void_ty(builder->getContext()), "llvm.bi.sl.waitcnt",
          ValueRange({wait_cnt}));
      builder->create<LLVM::CallIntrinsicOp>(
          op->getLoc(), void_ty(builder->getContext()),
          "llvm.bi.sl.barrier.alu", ValueRange({}));
    }
  }

  if (blockInfo->isIntersected(curBlockInfo)) {
    builder->setInsertionPoint(op);
    OpBuilder::InsertionGuard g(*builder);
    if (isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op)) {
      // TODO: sl_wait not work for no_sme, may ixcc problem
      auto asyncCopyGlobalToLocalOp =
          dyn_cast<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
      Value buffer_idx = asyncCopyGlobalToLocalOp.getOperand(1)
                             .getDefiningOp<triton::gpu::MemDescSubviewOp>()
                             .getOffsets()[0];
      if (auto constOp = buffer_idx.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          if (intAttr.getInt() == 0) {
            auto cnt_ty = builder->getIntegerType(64);
            Value wait_cnt = builder->create<LLVM::ConstantOp>(
                op->getLoc(), cnt_ty, IntegerAttr::get(cnt_ty, 12));
            builder->create<LLVM::CallIntrinsicOp>(
                op->getLoc(), void_ty(builder->getContext()),
                "llvm.bi.sl.waitcnt", ValueRange({wait_cnt}));
            builder->create<LLVM::CallIntrinsicOp>(
                op->getLoc(), void_ty(builder->getContext()),
                "llvm.bi.sl.barrier.alu", ValueRange({}));
            blockInfo->sync();
          }
        }
      }
      blockInfo->erase(curBlockInfo, 2);
    } else if (useBar) {
      auto barrierOp = builder->create<gpu::BarrierOp>(op->getLoc());
      blockInfo->sync();
    }
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}

} // namespace mlir
