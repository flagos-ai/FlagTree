//===- SynchronizationInsertion.cpp - Insert Pipeline Synchronization ----===//
//
// This file implements insertion of synchronization barriers for pipelined
// buffers, including producer-consumer coordination and async copy support.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/SynchronizationInsertion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "synchronization-insertion"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// SynchronizationInsertion Implementation
//===----------------------------------------------------------------------===//

void SynchronizationInsertion::insertSynchronization(
    PipelineOpportunity &opp, CircularBufferInfo &circularInfo,
    BufferAccessInfo *accessInfo) {
  // Main entry point for synchronization insertion
  LLVM_DEBUG(llvm::dbgs() << "Inserting synchronization for pipeline "
                          << circularInfo.pipelineId << "\n");

  scf::ForOp loop = circularInfo.loop;
  if (!loop) {
    LLVM_DEBUG(llvm::dbgs() << "No loop provided, skipping synchronization\n");
    return;
  }

  // Register this pipeline for potential synchronization fusion
  registerPipeline(circularInfo.pipelineId, circularInfo, opp);

  // NOTE: For Global→Shared pipelining with async copy, the synchronization
  // is handled directly by AsyncCommitGroupOp and AsyncWaitOp generated in
  // CircularBufferTransform::transformGlobalLoad. We skip the explicit
  // barrier insertion here to avoid generating invalid function calls.
  //
  // The fake func::CallOp to "triton_gpu.pipeline_*" functions were placeholders
  // that don't exist in Triton's dialect. Proper synchronization uses:
  // - triton::gpu::AsyncCopyGlobalToLocalOp for async copy
  // - triton::gpu::AsyncCommitGroupOp for committing transfers
  // - triton::gpu::AsyncWaitOp for waiting on completion

  if (circularInfo.useAsyncCopy) {
    // Async copy synchronization is handled by the transformation
    LLVM_DEBUG(llvm::dbgs() << "Async copy enabled - synchronization handled by transformation\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Synchronization insertion complete for pipeline "
                          << circularInfo.pipelineId << "\n");
}

void SynchronizationInsertion::registerPipeline(unsigned pipelineId,
                        CircularBufferInfo &circularInfo,
                        PipelineOpportunity &opp) {
  // Stub implementation
  PipelineInfo info;
  info.pipelineId = pipelineId;
  info.buffers.clear();
  info.buffers.push_back(circularInfo.originalBuffer);
  info.loop = opp.loop;
  info.numStages = circularInfo.numStages;
  info.scope = "shared";
  info.canFuseSync = false;

  pipelines[pipelineId] = info;
}

void SynchronizationInsertion::insertPipelineInit(
    CircularBufferInfo &info) {
  // Insert pipeline initialization before the loop
  scf::ForOp loop = info.loop;
  if (!loop) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loop);

  // Create a call to triton_gpu.pipeline_init
  // This serves as metadata for the pipeline initialization
  Location loc = loop.getLoc();
  auto noneType = builder.getType<mlir::NoneType>();

  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_init",
                               TypeRange{}, ValueRange{});

  LLVM_DEBUG(llvm::dbgs() << "Inserted pipeline init before loop\n");
}

void SynchronizationInsertion::insertPipelineFlush(
    CircularBufferInfo &info) {
  // Insert pipeline flush after the loop
  scf::ForOp loop = info.loop;
  if (!loop) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(loop);

  // Create a call to triton_gpu.pipeline_flush
  Location loc = loop.getLoc();

  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_flush",
                               TypeRange{}, ValueRange{});

  LLVM_DEBUG(llvm::dbgs() << "Inserted pipeline flush after loop\n");
}

void SynchronizationInsertion::insertProducerBarriers(Operation *producerOp,
                                                       unsigned pipelineId,
                                                       unsigned numStages) {
  if (!producerOp) {
    return;
  }

  // Insert producer-side barriers: acquire and commit
  OpBuilder::InsertionGuard guard(builder);
  Location loc = producerOp->getLoc();

  // Insert acquire before the producer operation
  builder.setInsertionPoint(producerOp);
  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_producer_acquire",
                               TypeRange{}, ValueRange{});

  // Insert commit after the producer operation
  builder.setInsertionPointAfter(producerOp);
  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_producer_commit",
                               TypeRange{}, ValueRange{});

  LLVM_DEBUG(llvm::dbgs() << "Inserted producer barriers for pipeline "
                          << pipelineId << "\n");
}

void SynchronizationInsertion::insertConsumerBarriers(Operation *consumerOp,
                                                       unsigned pipelineId,
                                                       unsigned numStages,
                                                       bool conditionalWait) {
  if (!consumerOp) {
    return;
  }

  // Insert consumer-side barriers: wait and release
  OpBuilder::InsertionGuard guard(builder);
  Location loc = consumerOp->getLoc();

  // Insert wait before the consumer operation
  builder.setInsertionPoint(consumerOp);
  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_consumer_wait",
                               TypeRange{}, ValueRange{});

  // Insert release after the consumer operation
  builder.setInsertionPointAfter(consumerOp);
  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_consumer_release",
                               TypeRange{}, ValueRange{});

  LLVM_DEBUG(llvm::dbgs() << "Inserted consumer barriers for pipeline "
                          << pipelineId << "\n");
}

void SynchronizationInsertion::insertConditionalConsumerWait(scf::ForOp loop,
                                                       unsigned pipelineId,
                                                       unsigned numStages,
                                                       CircularBufferInfo &info) {
  if (!loop) {
    return;
  }

  // Insert conditional consumer wait at the beginning of the loop body
  // This is used for chained pipelines where consumers need to wait
  // for producers from previous iterations
  OpBuilder::InsertionGuard guard(builder);
  Location loc = loop.getLoc();

  // Insert at the beginning of the loop body
  builder.setInsertionPointToStart(loop.getBody());

  // Create a conditional wait that checks iteration number
  // For iterations < numStages, we don't need to wait
  // For later iterations, wait for the data to be ready
  Value iv = loop.getInductionVar();

  // Create numStages constant with same type as induction variable
  Type ivType = iv.getType();
  Value numStagesConstant;
  if (ivType.isIndex()) {
    numStagesConstant = builder.create<arith::ConstantIndexOp>(loc, numStages);
  } else {
    // Assume integer type (typically i32)
    numStagesConstant = builder.create<arith::ConstantOp>(
        loc, ivType, builder.getIntegerAttr(ivType, numStages));
  }

  // Create condition: iv < numStages
  Value condition = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, iv, numStagesConstant);

  // Create if-then-else for conditional wait
  auto ifOp = builder.create<scf::IfOp>(loc, condition,
                                        /*hasElse=*/false);

  // In the else branch (when iv >= numStages), insert the wait
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<func::CallOp>(loc, "triton_gpu.pipeline_consumer_wait",
                               TypeRange{}, ValueRange{});

  LLVM_DEBUG(llvm::dbgs() << "Inserted conditional consumer wait for pipeline "
                          << pipelineId << "\n");
}

void SynchronizationInsertion::insertAsyncCopy(Operation *storeOp,
                                                CircularBufferInfo &info) {
  if (!storeOp) {
    return;
  }

  // Insert async copy intrinsic for global to shared memory transfers
  // This will be lowered to cp.async on NVIDIA Ampere+ or load+store otherwise
  OpBuilder::InsertionGuard guard(builder);
  Location loc = storeOp->getLoc();

  // Insert async copy call before the store operation
  builder.setInsertionPoint(storeOp);
  builder.create<func::CallOp>(loc, "triton_gpu.async_copy_global_to_shared",
                               TypeRange{}, ValueRange{});

  LLVM_DEBUG(llvm::dbgs() << "Inserted async copy intrinsic\n");
}

bool SynchronizationInsertion::canShareSynchronization(
    const PipelineInfo &pipeline1, const PipelineInfo &pipeline2) {
  // Check if two pipelines can share synchronization barriers
  // This reduces barrier overhead when multiple buffers are in the same pipeline

  // Must be in the same loop
  if (pipeline1.loop != pipeline2.loop) {
    return false;
  }

  // Must have the same number of stages
  if (pipeline1.numStages != pipeline2.numStages) {
    return false;
  }

  // Must be in the same memory scope
  if (pipeline1.scope != pipeline2.scope) {
    return false;
  }

  // Buffers in the same memory scope and loop can share synchronization
  return true;
}

bool SynchronizationInsertion::canFuseSynchronization(
    ArrayRef<Value> buffers, BufferAccessAnalysis &analysis) {
  // Check if multiple buffers can share synchronization barriers
  if (buffers.size() <= 1) {
    return false;
  }

  // Get the first buffer's access info
  BufferAccessInfo *firstInfo = analysis.getAccessInfo(buffers[0]);
  if (!firstInfo) {
    return false;
  }

  // Check if all buffers have compatible access patterns
  for (size_t i = 1; i < buffers.size(); ++i) {
    BufferAccessInfo *currentInfo = analysis.getAccessInfo(buffers[i]);
    if (!currentInfo) {
      return false;
    }

    // Buffers must be in the same memory scope
    if (currentInfo->scope != firstInfo->scope) {
      return false;
    }

    // Buffers must be in the same loop context
    if (currentInfo->loopContext != firstInfo->loopContext) {
      return false;
    }
  }

  return true;
}

void SynchronizationInsertion::insertFusedSynchronization(
    CircularBufferInfo &info, BufferAccessInfo *accessInfo) {
  // Insert shared synchronization for multiple buffers
  LLVM_DEBUG(llvm::dbgs() << "Inserting fused synchronization\n");

  // For now, just use the same synchronization as individual
  // The fusion happens because we share the pipeline ID
  insertPipelineInit(info);
  insertPipelineFlush(info);

  if (accessInfo && accessInfo->producer) {
    insertProducerBarriers(accessInfo->producer, info.pipelineId,
                           info.numStages);
  }

  if (accessInfo && !accessInfo->consumers.empty()) {
    for (Operation *consumer : accessInfo->consumers) {
      insertConsumerBarriers(consumer, info.pipelineId,
                            info.numStages, false);
    }
  }
}

void SynchronizationInsertion::insertIndividualSynchronization(
    CircularBufferInfo &info, BufferAccessInfo *accessInfo) {
  // Insert individual synchronization per buffer
  LLVM_DEBUG(llvm::dbgs() << "Inserting individual synchronization\n");

  insertPipelineInit(info);
  insertPipelineFlush(info);

  if (accessInfo && accessInfo->producer) {
    insertProducerBarriers(accessInfo->producer, info.pipelineId,
                           info.numStages);
  }

  if (accessInfo && !accessInfo->consumers.empty()) {
    bool needsConditionalWait = info.numStages > 2;
    for (Operation *consumer : accessInfo->consumers) {
      insertConsumerBarriers(consumer, info.pipelineId,
                            info.numStages, needsConditionalWait);
    }

    if (needsConditionalWait) {
      insertConditionalConsumerWait(info.loop, info.pipelineId,
                                    info.numStages, info);
    }
  }

  if (info.useAsyncCopy && accessInfo && accessInfo->producer) {
    insertAsyncCopy(accessInfo->producer, info);
  }
}
