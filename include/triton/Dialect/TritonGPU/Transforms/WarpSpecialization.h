#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineOpportunityDetector.h"
#include "triton/Dialect/TritonGPU/Transforms/CircularBufferTransform.h"

namespace mlir {
namespace triton {
namespace gpu {

/// Warp role in specialized execution
enum class WarpRole {
  Producer,  // Loads data from global memory
  Consumer,  // Performs computation
  Mixed      // Both load and compute (default)
};

/// Configuration for warp specialization
struct WarpSpecializationConfig {
  /// Number of producer warps (for data loading)
  unsigned numProducerWarps = 1;

  /// Number of consumer warps (for computation)
  unsigned numConsumerWarps = 3;

  /// Total number of warps (typically 4 for 128-thread blocks)
  unsigned totalWarps = 4;

  /// Whether to use persistent producer warps
  bool persistentProducers = true;

  /// Whether to enable double buffering for producer warps
  bool doubleBuffer = true;

  /// Minimum elements per producer warp for efficiency
  unsigned minElementsPerProducer = 256;

  WarpSpecializationConfig() = default;
};

/// Information about warp specialization transformation
struct WarpSpecializationInfo {
  /// The loop being specialized
  scf::ForOp loop;

  /// Configuration used
  WarpSpecializationConfig config;

  /// Producer operations (moved to producer warps)
  SmallVector<Operation *> producerOps;

  /// Consumer operations (moved to consumer warps)
  SmallVector<Operation *> consumerOps;

  /// Warp ID value (computed from thread ID)
  Value warpId;

  /// Whether this warp is a producer
  Value isProducerWarp;

  /// Whether this warp is a consumer
  Value isConsumerWarp;

  /// Associated pipeline ID
  unsigned pipelineId = 0;

  WarpSpecializationInfo() = default;
};

/// Warp Specialization transformer for advanced pipelining
///
/// This class implements warp specialization where:
/// - Producer warps are dedicated to loading data from global memory
/// - Consumer warps are dedicated to computation (e.g., matrix multiply)
/// - Proper synchronization ensures correctness
///
/// Benefits:
/// - Better memory latency hiding
/// - Reduced register pressure per warp
/// - Improved occupancy
///
class WarpSpecialization {
public:
  explicit WarpSpecialization(OpBuilder &builder) : builder(builder) {}

  /// Check if warp specialization is beneficial for the given opportunity
  bool isProfitable(const PipelineOpportunity &opp,
                    const CircularBufferInfo &circularInfo);

  /// Analyze loop to determine optimal warp configuration
  WarpSpecializationConfig analyzeLoop(scf::ForOp loop,
                                        const PipelineOpportunity &opp);

  /// Apply warp specialization transformation
  WarpSpecializationInfo apply(const PipelineOpportunity &opp,
                                CircularBufferInfo &circularInfo,
                                unsigned pipelineId);

  /// Insert warp-level synchronization barriers
  void insertWarpBarriers(WarpSpecializationInfo &info);

  /// Get the current warp ID value (creates if not exists)
  Value getWarpId(Location loc);

  /// Create predicate for producer warp check
  Value createProducerPredicate(Location loc, Value warpId,
                                 const WarpSpecializationConfig &config);

  /// Create predicate for consumer warp check
  Value createConsumerPredicate(Location loc, Value warpId,
                                 const WarpSpecializationConfig &config);

private:
  OpBuilder &builder;

  /// Cached warp ID value
  Value cachedWarpId;

  /// Partition operations into producer and consumer sets
  void partitionOperations(scf::ForOp loop,
                           SmallVector<Operation *> &producerOps,
                           SmallVector<Operation *> &consumerOps);

  /// Move producer operations under producer predicate
  void moveProducerOps(WarpSpecializationInfo &info);

  /// Move consumer operations under consumer predicate
  void moveConsumerOps(WarpSpecializationInfo &info);

  /// Estimate producer work (memory operations)
  unsigned estimateProducerWork(scf::ForOp loop);

  /// Estimate consumer work (compute operations)
  unsigned estimateConsumerWork(scf::ForOp loop);

  /// Create warp-level barrier
  void createWarpBarrier(Location loc);
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_H
