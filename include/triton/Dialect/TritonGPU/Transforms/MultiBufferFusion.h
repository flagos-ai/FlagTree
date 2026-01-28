#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_MULTIBUFFERFUSION_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_MULTIBUFFERFUSION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineOpportunityDetector.h"
#include "triton/Dialect/TritonGPU/Transforms/CircularBufferTransform.h"

namespace mlir {
namespace triton {
namespace gpu {

/// Information about a buffer group for fusion
struct BufferGroup {
  /// Buffers in this group
  SmallVector<Value> buffers;

  /// Common loop context
  scf::ForOp loop;

  /// Shared pipeline stages
  unsigned numStages;

  /// Use shared synchronization
  bool sharedSync = true;

  /// Producer operations for all buffers
  SmallVector<Operation *> producers;

  /// Consumer operations for all buffers
  SmallVector<Operation *> consumers;

  BufferGroup() : numStages(1) {}
};

/// Information about multi-buffer fusion transformation
struct MultiBufferFusionInfo {
  /// The loop being transformed
  scf::ForOp loop;

  /// Groups of buffers that share synchronization
  SmallVector<BufferGroup> groups;

  /// Shared barrier for the fused buffers
  Value sharedBarrier;

  /// Pipeline ID
  unsigned pipelineId = 0;

  MultiBufferFusionInfo() = default;
};

/// Multi-buffer Fusion for efficient synchronization sharing
///
/// This class implements multi-buffer fusion which allows multiple
/// buffers (e.g., K and V in attention) to share synchronization
/// barriers when they have similar access patterns.
///
/// Benefits:
/// - Reduced barrier overhead (fewer sync points)
/// - Better latency hiding
/// - Simplified control flow
///
class MultiBufferFusion {
public:
  explicit MultiBufferFusion(OpBuilder &builder) : builder(builder) {}

  /// Find groups of buffers that can share synchronization
  SmallVector<BufferGroup> findFusionGroups(
      const SmallVector<PipelineOpportunity> &opportunities);

  /// Check if two buffers can share synchronization
  bool canFuse(const PipelineOpportunity &a, const PipelineOpportunity &b);

  /// Apply multi-buffer fusion to a group
  MultiBufferFusionInfo apply(BufferGroup &group, unsigned pipelineId);

  /// Create shared synchronization for a buffer group
  void createSharedSync(MultiBufferFusionInfo &info);

  /// Merge producer operations from multiple buffers
  void mergeProducers(BufferGroup &group, MultiBufferFusionInfo &info);

  /// Merge consumer operations from multiple buffers
  void mergeConsumers(BufferGroup &group, MultiBufferFusionInfo &info);

private:
  OpBuilder &builder;

  /// Check if operations are in the same loop
  bool sameLoop(Operation *a, Operation *b);

  /// Check if buffers have compatible access patterns
  bool compatibleAccess(const PipelineOpportunity &a,
                        const PipelineOpportunity &b);

  /// Estimate benefit of fusion
  double estimateFusionBenefit(const BufferGroup &group);
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_MULTIBUFFERFUSION_H
