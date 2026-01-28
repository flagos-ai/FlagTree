//===- WarpSpecialization.cpp - Warp Specialization for Pipelining --------===//
//
// This file implements warp specialization optimization where producer warps
// are dedicated to loading data and consumer warps to computation.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "warp-specialization"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// WarpSpecialization Implementation
//===----------------------------------------------------------------------===//

bool WarpSpecialization::isProfitable(const PipelineOpportunity &opp,
                                       const CircularBufferInfo &circularInfo) {
  if (!circularInfo.loop) {
    return false;
  }

  // Check if the loop has enough work for specialization
  unsigned producerWork = estimateProducerWork(circularInfo.loop);
  unsigned consumerWork = estimateConsumerWork(circularInfo.loop);

  // Warp specialization is beneficial when:
  // 1. There's significant producer work (memory operations)
  // 2. There's significant consumer work (compute operations)
  // 3. The ratio allows good overlap

  if (producerWork < 10 || consumerWork < 20) {
    LLVM_DEBUG(llvm::dbgs() << "Warp specialization not profitable: "
                            << "producerWork=" << producerWork
                            << ", consumerWork=" << consumerWork << "\n");
    return false;
  }

  // Check for minimum pipeline stages
  if (circularInfo.numStages < 2) {
    LLVM_DEBUG(llvm::dbgs() << "Warp specialization requires >= 2 stages\n");
    return false;
  }

  // Check for DotOp presence (matmul kernels benefit most)
  bool hasDotOp = false;
  scf::ForOp loop = circularInfo.loop;
  if (loop) {
    loop.getBody()->walk([&](triton::DotOp dotOp) {
      hasDotOp = true;
    });
  }

  if (!hasDotOp) {
    LLVM_DEBUG(llvm::dbgs() << "Warp specialization most beneficial for matmul kernels\n");
    // Still allow but with reduced confidence
  }

  double ratio = static_cast<double>(producerWork) / consumerWork;
  LLVM_DEBUG(llvm::dbgs() << "Warp specialization analysis: "
                          << "producer/consumer ratio=" << ratio
                          << ", hasDotOp=" << hasDotOp << "\n");

  // Profitable if ratio is reasonable (not too imbalanced)
  return ratio >= 0.1 && ratio <= 2.0;
}

WarpSpecializationConfig WarpSpecialization::analyzeLoop(
    scf::ForOp loop, const PipelineOpportunity &opp) {

  WarpSpecializationConfig config;

  if (!loop) {
    return config;
  }

  // Estimate work distribution
  unsigned producerWork = estimateProducerWork(loop);
  unsigned consumerWork = estimateConsumerWork(loop);

  // Total warps based on typical block configuration
  // Assuming BLOCK_SIZE=128 threads = 4 warps (32 threads/warp)
  config.totalWarps = 4;

  // Allocate warps based on work ratio
  double ratio = static_cast<double>(producerWork) /
                 (producerWork + consumerWork);

  if (ratio < 0.2) {
    // Light producer work - 1 producer, 3 consumers
    config.numProducerWarps = 1;
    config.numConsumerWarps = 3;
  } else if (ratio < 0.4) {
    // Moderate producer work - 1 producer, 3 consumers
    config.numProducerWarps = 1;
    config.numConsumerWarps = 3;
  } else {
    // Heavy producer work - 2 producers, 2 consumers
    config.numProducerWarps = 2;
    config.numConsumerWarps = 2;
  }

  // Enable double buffering for better overlap
  config.doubleBuffer = (opp.numStages >= 2);

  // Persistent producers help with large loops
  // Check if the loop has a large constant trip count
  config.persistentProducers = true;
  auto upperBound = loop.getUpperBound();
  auto lowerBound = loop.getLowerBound();
  auto step = loop.getStep();
  if (auto ubConst = upperBound.getDefiningOp<arith::ConstantOp>()) {
    if (auto lbConst = lowerBound.getDefiningOp<arith::ConstantOp>()) {
      if (auto stepConst = step.getDefiningOp<arith::ConstantOp>()) {
        auto ubInt = mlir::dyn_cast<IntegerAttr>(ubConst.getValue());
        auto lbInt = mlir::dyn_cast<IntegerAttr>(lbConst.getValue());
        auto stepInt = mlir::dyn_cast<IntegerAttr>(stepConst.getValue());
        if (ubInt && lbInt && stepInt && stepInt.getInt() > 0) {
          int64_t extent = (ubInt.getInt() - lbInt.getInt()) / stepInt.getInt();
          config.persistentProducers = extent >= 8;
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Warp configuration: "
                          << config.numProducerWarps << " producers, "
                          << config.numConsumerWarps << " consumers"
                          << ", doubleBuffer=" << config.doubleBuffer
                          << ", persistent=" << config.persistentProducers << "\n");

  return config;
}

WarpSpecializationInfo WarpSpecialization::apply(
    const PipelineOpportunity &opp, CircularBufferInfo &circularInfo,
    unsigned pipelineId) {

  WarpSpecializationInfo info;
  info.loop = circularInfo.loop;
  info.pipelineId = pipelineId;

  if (!info.loop) {
    return info;
  }

  // Analyze and configure
  info.config = analyzeLoop(info.loop, opp);

  // Partition operations
  partitionOperations(info.loop, info.producerOps, info.consumerOps);

  // Get warp ID
  Location loc = info.loop.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(info.loop.getBody());

  info.warpId = getWarpId(loc);

  // Create predicates
  info.isProducerWarp = createProducerPredicate(loc, info.warpId, info.config);
  info.isConsumerWarp = createConsumerPredicate(loc, info.warpId, info.config);

  // Move operations under predicates
  moveProducerOps(info);
  moveConsumerOps(info);

  // Insert barriers
  insertWarpBarriers(info);

  LLVM_DEBUG(llvm::dbgs() << "Applied warp specialization: "
                          << info.producerOps.size() << " producer ops, "
                          << info.consumerOps.size() << " consumer ops\n");

  return info;
}

Value WarpSpecialization::getWarpId(Location loc) {
  if (cachedWarpId) {
    return cachedWarpId;
  }

  // Get thread ID and compute warp ID
  // warpId = threadId / 32

  // Create thread ID (using GPU thread ID intrinsic)
  // In Triton, this is typically available as a program_id or computed
  Value threadId = builder.create<triton::GetProgramIdOp>(
      loc, builder.getI32Type(), triton::ProgramIDDim::X);

  // For warp specialization within a block, we need the lane-level thread ID
  // This is typically computed as: local_thread_id = global_thread_id % BLOCK_SIZE
  // Then: warp_id = local_thread_id / 32

  // Create constants
  Value warpSize = builder.create<arith::ConstantOp>(
      loc, builder.getI32Type(),
      builder.getI32IntegerAttr(32));

  // Compute warp ID
  cachedWarpId = builder.create<arith::DivUIOp>(loc, threadId, warpSize);

  return cachedWarpId;
}

Value WarpSpecialization::createProducerPredicate(
    Location loc, Value warpId, const WarpSpecializationConfig &config) {

  // Producer warps are warpId < numProducerWarps
  Value numProducers = builder.create<arith::ConstantOp>(
      loc, warpId.getType(),
      builder.getI32IntegerAttr(config.numProducerWarps));

  return builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, warpId, numProducers);
}

Value WarpSpecialization::createConsumerPredicate(
    Location loc, Value warpId, const WarpSpecializationConfig &config) {

  // Consumer warps are warpId >= numProducerWarps
  Value numProducers = builder.create<arith::ConstantOp>(
      loc, warpId.getType(),
      builder.getI32IntegerAttr(config.numProducerWarps));

  return builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, warpId, numProducers);
}

void WarpSpecialization::partitionOperations(
    scf::ForOp loop, SmallVector<Operation *> &producerOps,
    SmallVector<Operation *> &consumerOps) {

  // Classify operations as producer or consumer
  loop.getBody()->walk([&](Operation *op) {
    // Skip terminators
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return;
    }

    // Producer operations: memory loads
    if (isa<triton::LoadOp>(op) ||
        isa<triton::gpu::LocalStoreOp>(op)) {
      producerOps.push_back(op);
      return;
    }

    // Consumer operations: computation
    if (isa<triton::DotOp>(op) ||
        isa<triton::gpu::LocalLoadOp>(op) ||
        isa<triton::StoreOp>(op) ||
        isa<arith::MulFOp>(op) ||
        isa<arith::AddFOp>(op)) {
      consumerOps.push_back(op);
      return;
    }

    // Default: treat as consumer (compute) operation
    consumerOps.push_back(op);
  });

  LLVM_DEBUG(llvm::dbgs() << "Partitioned: " << producerOps.size()
                          << " producer ops, " << consumerOps.size()
                          << " consumer ops\n");
}

void WarpSpecialization::moveProducerOps(WarpSpecializationInfo &info) {
  if (info.producerOps.empty() || !info.isProducerWarp) {
    return;
  }

  // Wrap producer operations in an if (isProducerWarp) block
  Location loc = info.loop.getLoc();

  for (Operation *op : info.producerOps) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);

    // Create scf.if for producer predicate
    auto ifOp = builder.create<scf::IfOp>(
        loc, info.isProducerWarp,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          // Clone operation inside the if block
          thenBuilder.clone(*op);
          thenBuilder.create<scf::YieldOp>(thenLoc);
        });

    // Mark original for deletion
    op->setAttr("warp_specialized", builder.getUnitAttr());
  }

  LLVM_DEBUG(llvm::dbgs() << "Moved " << info.producerOps.size()
                          << " ops to producer warps\n");
}

void WarpSpecialization::moveConsumerOps(WarpSpecializationInfo &info) {
  // Consumer ops typically don't need explicit predication
  // as they use data from shared memory which all warps can access

  // However, we can optimize by having consumers skip
  // iteration while waiting for producers

  LLVM_DEBUG(llvm::dbgs() << "Consumer ops remain accessible to all warps\n");
}

void WarpSpecialization::insertWarpBarriers(WarpSpecializationInfo &info) {
  if (!info.loop) {
    return;
  }

  // Insert barrier after producer operations
  // This synchronizes producer and consumer warps

  Location loc = info.loop.getLoc();

  // Find insertion point after producers, before consumers
  Operation *lastProducer = nullptr;
  for (Operation *op : info.producerOps) {
    if (!lastProducer || op->isBeforeInBlock(lastProducer)) {
      lastProducer = op;
    }
  }

  if (lastProducer) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(lastProducer);
    createWarpBarrier(loc);
  }

  LLVM_DEBUG(llvm::dbgs() << "Inserted warp barriers for synchronization\n");
}

void WarpSpecialization::createWarpBarrier(Location loc) {
  // Create a GPU barrier for warp synchronization
  // In Triton, this maps to __syncthreads() / bar.sync

  builder.create<::mlir::gpu::BarrierOp>(loc);

  LLVM_DEBUG(llvm::dbgs() << "Created warp barrier\n");
}

unsigned WarpSpecialization::estimateProducerWork(scf::ForOp loop) {
  unsigned work = 0;

  loop.getBody()->walk([&](Operation *op) {
    if (isa<triton::LoadOp>(op)) {
      work += 10;  // Global load is expensive
    } else if (isa<triton::gpu::LocalStoreOp>(op)) {
      work += 2;   // Shared memory store
    } else if (isa<triton::AdvanceOp>(op)) {
      work += 1;   // Pointer arithmetic
    }
  });

  return work;
}

unsigned WarpSpecialization::estimateConsumerWork(scf::ForOp loop) {
  unsigned work = 0;

  loop.getBody()->walk([&](Operation *op) {
    if (isa<triton::DotOp>(op)) {
      work += 50;  // Matrix multiply is heavy compute
    } else if (isa<triton::gpu::LocalLoadOp>(op)) {
      work += 2;   // Shared memory load
    } else if (isa<arith::MulFOp, arith::AddFOp>(op)) {
      work += 1;   // Simple arithmetic
    } else if (isa<triton::StoreOp>(op)) {
      work += 5;   // Global store
    }
  });

  return work;
}
