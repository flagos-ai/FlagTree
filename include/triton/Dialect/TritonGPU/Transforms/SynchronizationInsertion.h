#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_SYNCHRONIZATIONINSERTION_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_SYNCHRONIZATIONINSERTION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/Transforms/CircularBufferTransform.h"
#include "triton/Dialect/TritonGPU/Transforms/BufferAccessAnalysis.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace triton {
namespace gpu{

/// Pipeline metadata for tracking related buffers and synchronization
struct PipelineInfo {
  /// Unique pipeline identifier
  unsigned pipelineId;

  /// All buffers in this pipeline
  SmallVector<Value> buffers;

  /// Pipelined loop
  scf::ForOp loop;

  /// Number of stages
  unsigned numStages;

  /// Memory scope ("shared", "global", etc.)
  StringRef scope;

  /// Whether buffers can share synchronization
  bool canFuseSync;

  PipelineInfo()
      : pipelineId(0), loop(nullptr), numStages(1), scope(""),
        canFuseSync(false) {}
};

/// Insert synchronization barriers for pipelined buffers
class SynchronizationInsertion {
public:
  explicit SynchronizationInsertion(OpBuilder &builder) : builder(builder) {}

  /// Insert all synchronization for a pipelined buffer
  void insertSynchronization(PipelineOpportunity &opp,
                             CircularBufferInfo &circularInfo,
                             BufferAccessInfo *accessInfo);

  /// Register a pipeline for potential synchronization fusion
  void registerPipeline(unsigned pipelineId,
                        CircularBufferInfo &circularInfo,
                        PipelineOpportunity &opp);

private:
  OpBuilder &builder;

  /// Registered pipelines
  DenseMap<unsigned, PipelineInfo> pipelines;

  /// Insert pipeline initialization (before loop)
  void insertPipelineInit(CircularBufferInfo &info);

  /// Insert pipeline flush (after loop)
  void insertPipelineFlush(CircularBufferInfo &info);

  /// Insert producer-side barriers (acquire, commit)
  void insertProducerBarriers(Operation *producerOp, unsigned pipelineId,
                              unsigned numStages);

  /// Insert consumer-side barriers (wait, release)
  void insertConsumerBarriers(Operation *consumerOp, unsigned pipelineId,
                              unsigned numStages, bool conditionalWait);

  /// Insert conditional consumer wait for chained pipelines
  void insertConditionalConsumerWait(scf::ForOp loop, unsigned pipelineId,
                                     unsigned numStages,
                                     CircularBufferInfo &info);

  /// Insert async memory copy intrinsic
  void insertAsyncCopy(Operation *storeOp, CircularBufferInfo &info);

  /// Check if multiple buffers can share synchronization
  bool canFuseSynchronization(ArrayRef<Value> buffers,
                              BufferAccessAnalysis &analysis);

  /// Insert fused synchronization for multiple buffers
  void insertFusedSynchronization(CircularBufferInfo &info,
                                  BufferAccessInfo *accessInfo);

  /// Insert individual synchronization per buffer
  void insertIndividualSynchronization(CircularBufferInfo &info,
                                       BufferAccessInfo *accessInfo);

  /// Check if two pipelines can share synchronization
  bool canShareSynchronization(const PipelineInfo &pipeline1,
                               const PipelineInfo &pipeline2);
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_SYNCHRONIZATIONINSERTION_H
