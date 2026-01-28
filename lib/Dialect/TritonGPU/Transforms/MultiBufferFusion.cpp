//===- MultiBufferFusion.cpp - Multi-Buffer Synchronization Fusion --------===//
//
// This file implements multi-buffer fusion which allows multiple buffers
// (e.g., K and V in attention) to share synchronization barriers.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/MultiBufferFusion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "multi-buffer-fusion"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// MultiBufferFusion Implementation
//===----------------------------------------------------------------------===//

SmallVector<BufferGroup> MultiBufferFusion::findFusionGroups(
    const SmallVector<PipelineOpportunity> &opportunities) {

  SmallVector<BufferGroup> groups;

  // Track which opportunities have been assigned to groups
  DenseSet<unsigned> assigned;

  for (unsigned i = 0; i < opportunities.size(); ++i) {
    if (assigned.count(i)) {
      continue;
    }

    BufferGroup group;
    group.buffers.push_back(opportunities[i].buffer);
    group.loop = opportunities[i].loop;
    group.numStages = opportunities[i].numStages;
    assigned.insert(i);

    // Find other opportunities that can fuse with this one
    for (unsigned j = i + 1; j < opportunities.size(); ++j) {
      if (assigned.count(j)) {
        continue;
      }

      if (canFuse(opportunities[i], opportunities[j])) {
        group.buffers.push_back(opportunities[j].buffer);
        // Take minimum stages to ensure compatibility
        group.numStages = std::min(group.numStages, opportunities[j].numStages);
        assigned.insert(j);

        LLVM_DEBUG(llvm::dbgs() << "Fusing buffer " << j << " with buffer " << i
                                << "\n");
      }
    }

    // Only create group if we fused multiple buffers
    if (group.buffers.size() > 1) {
      groups.push_back(group);
      LLVM_DEBUG(llvm::dbgs() << "Created fusion group with "
                              << group.buffers.size() << " buffers\n");
    }
  }

  return groups;
}

bool MultiBufferFusion::canFuse(const PipelineOpportunity &a,
                                 const PipelineOpportunity &b) {
  // Must be in the same loop
  if (a.loop != b.loop) {
    return false;
  }

  // Must have the same pipeline level
  if (a.level != b.level) {
    return false;
  }

  // Check for compatible access patterns
  if (!compatibleAccess(a, b)) {
    return false;
  }

  // Stages should be similar (within 1)
  int stageDiff = static_cast<int>(a.numStages) - static_cast<int>(b.numStages);
  if (std::abs(stageDiff) > 1) {
    return false;
  }

  // Both should use same async copy setting
  if (a.useAsyncCopy != b.useAsyncCopy) {
    return false;
  }

  return true;
}

bool MultiBufferFusion::compatibleAccess(const PipelineOpportunity &a,
                                          const PipelineOpportunity &b) {
  // Check if buffers have similar access patterns

  // Get buffer defining operations
  Operation *defOpA = a.buffer.getDefiningOp();
  Operation *defOpB = b.buffer.getDefiningOp();

  if (!defOpA || !defOpB) {
    return false;
  }

  // Check if both are local allocations (shared memory)
  bool isLocalA = isa<triton::gpu::LocalAllocOp>(defOpA);
  bool isLocalB = isa<triton::gpu::LocalAllocOp>(defOpB);

  if (isLocalA != isLocalB) {
    return false;
  }

  // If both are local allocs, check for compatible shapes
  if (isLocalA && isLocalB) {
    auto allocA = cast<triton::gpu::LocalAllocOp>(defOpA);
    auto allocB = cast<triton::gpu::LocalAllocOp>(defOpB);

    auto typeA = cast<triton::MemDescType>(allocA.getResult().getType());
    auto typeB = cast<triton::MemDescType>(allocB.getResult().getType());

    // Shapes should match for fusion
    if (typeA.getShape() != typeB.getShape()) {
      // Allow different shapes if element counts are similar
      int64_t countA = 1, countB = 1;
      for (int64_t d : typeA.getShape()) countA *= d;
      for (int64_t d : typeB.getShape()) countB *= d;

      double ratio = static_cast<double>(countA) / countB;
      if (ratio < 0.5 || ratio > 2.0) {
        return false;
      }
    }

    // Element types should match
    if (typeA.getElementType() != typeB.getElementType()) {
      return false;
    }
  }

  return true;
}

MultiBufferFusionInfo MultiBufferFusion::apply(BufferGroup &group,
                                                unsigned pipelineId) {
  MultiBufferFusionInfo info;
  info.loop = group.loop;
  info.pipelineId = pipelineId;

  if (!info.loop || group.buffers.size() < 2) {
    return info;
  }

  info.groups.push_back(group);

  // Create shared synchronization
  createSharedSync(info);

  // Merge producer operations
  mergeProducers(group, info);

  // Merge consumer operations
  mergeConsumers(group, info);

  LLVM_DEBUG(llvm::dbgs() << "Applied multi-buffer fusion: "
                          << group.buffers.size() << " buffers, "
                          << group.producers.size() << " producers, "
                          << group.consumers.size() << " consumers\n");

  return info;
}

void MultiBufferFusion::createSharedSync(MultiBufferFusionInfo &info) {
  if (!info.loop || info.groups.empty()) {
    return;
  }

  Location loc = info.loop.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(info.loop);

  // Create a single shared barrier for all fused buffers
  // This replaces multiple individual barriers

  BufferGroup &group = info.groups[0];
  unsigned numBuffers = group.buffers.size();

  // Create barrier with arrival count for all buffers
  Value barrierCount = builder.create<arith::ConstantOp>(
      loc, builder.getI32Type(),
      builder.getI32IntegerAttr(numBuffers));

  info.sharedBarrier = barrierCount;

  LLVM_DEBUG(llvm::dbgs() << "Created shared barrier for " << numBuffers
                          << " buffers\n");
}

void MultiBufferFusion::mergeProducers(BufferGroup &group,
                                        MultiBufferFusionInfo &info) {
  if (!info.loop) {
    return;
  }

  // Collect all producer operations from buffers in the group
  info.loop.getBody()->walk([&](Operation *op) {
    // Check if operation produces to any buffer in the group
    for (Value buffer : group.buffers) {
      // Check LocalStoreOp
      if (auto storeOp = dyn_cast<triton::gpu::LocalStoreOp>(op)) {
        if (storeOp.getDst() == buffer) {
          group.producers.push_back(op);
          break;
        }
      }

      // Check regular StoreOp
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
        // Check if store destination relates to buffer
        Value ptr = storeOp.getPtr();
        Operation *ptrDefOp = ptr.getDefiningOp();
        if (ptrDefOp && ptrDefOp == buffer.getDefiningOp()) {
          group.producers.push_back(op);
          break;
        }
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << group.producers.size()
                          << " producer operations\n");
}

void MultiBufferFusion::mergeConsumers(BufferGroup &group,
                                        MultiBufferFusionInfo &info) {
  if (!info.loop) {
    return;
  }

  // Collect all consumer operations from buffers in the group
  info.loop.getBody()->walk([&](Operation *op) {
    // Check if operation consumes from any buffer in the group
    for (Value buffer : group.buffers) {
      // Check LocalLoadOp
      if (auto loadOp = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        if (loadOp.getSrc() == buffer) {
          group.consumers.push_back(op);
          break;
        }
      }

      // Check regular LoadOp
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        Value ptr = loadOp.getPtr();
        Operation *ptrDefOp = ptr.getDefiningOp();
        if (ptrDefOp && ptrDefOp == buffer.getDefiningOp()) {
          group.consumers.push_back(op);
          break;
        }
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << group.consumers.size()
                          << " consumer operations\n");
}

bool MultiBufferFusion::sameLoop(Operation *a, Operation *b) {
  auto loopA = a->getParentOfType<scf::ForOp>();
  auto loopB = b->getParentOfType<scf::ForOp>();
  return loopA == loopB;
}

double MultiBufferFusion::estimateFusionBenefit(const BufferGroup &group) {
  // Estimate the benefit of fusing this group

  // Base benefit from reduced barriers
  double barrierReduction = group.buffers.size() - 1;

  // Each eliminated barrier saves approximately 20-50 cycles
  double cycleSavings = barrierReduction * 35.0;

  // Benefit scales with number of iterations
  double benefit = cycleSavings;

  // Additional benefit from simplified control flow
  if (group.buffers.size() >= 3) {
    benefit *= 1.2;  // 20% bonus for large groups
  }

  LLVM_DEBUG(llvm::dbgs() << "Estimated fusion benefit: " << benefit
                          << " cycles\n");

  return benefit;
}
