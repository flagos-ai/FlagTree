#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_CIRCULARBUFFERTRANSFORM_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_CIRCULARBUFFERTRANSFORM_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineOpportunityDetector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace triton {
namespace gpu {

/// Information about a circular buffer transformation
struct CircularBufferInfo {
  /// Original buffer allocation
  Value originalBuffer;

  /// New circular buffer (expanded with stage dimension)
  Value circularBuffer;

  /// Number of pipeline stages
  unsigned numStages;

  /// Stride in elements between stages
  int64_t stride;

  /// Loop being pipelined
  scf::ForOp loop;

  /// Associated pipeline ID
  unsigned pipelineId;

  /// Use async copy intrinsics
  bool useAsyncCopy;

  /// Use swizzled indexing
  bool useSwizzle;

  CircularBufferInfo()
      : originalBuffer(nullptr), circularBuffer(nullptr), numStages(1),
        stride(0), loop(nullptr), pipelineId(0), useAsyncCopy(false),
        useSwizzle(false) {}
};

/// Transform buffer allocations and accesses to use circular buffering
class CircularBufferTransform {
public:
  explicit CircularBufferTransform(OpBuilder &builder) : builder(builder) {}

  /// Transform a buffer allocation to circular buffer
  CircularBufferInfo transformAllocation(const PipelineOpportunity &opp,
                                         unsigned pipelineId);

  /// Transform a store operation to use circular buffer indexing
  void transformStore(Operation *storeOp, CircularBufferInfo &info);

  /// Transform a load operation to use circular buffer indexing
  void transformLoad(Operation *loadOp, CircularBufferInfo &info);

  /// Transform a LocalStoreOp (Global→Shared or Register→Shared)
  void transformLocalStore(Operation *localStoreOp, CircularBufferInfo &info);

  /// Transform a LocalLoadOp (Shared→Register) for register pipelining
  void transformLocalLoad(Operation *localLoadOp, CircularBufferInfo &info);

  /// Transform a global LoadOp to use async copy (Global→Shared pipelining)
  /// This is the key method that generates cp.async operations
  void transformGlobalLoad(triton::LoadOp loadOp, CircularBufferInfo &info,
                           Value insertIdx, Value extractIdx);

  /// Allocate shared memory buffer for a load operation
  Value allocateSharedBuffer(triton::LoadOp loadOp, unsigned numStages);

  /// Get appropriate shared encoding for a load type
  Attribute getSharedEncodingForLoad(triton::LoadOp loadOp);

private:
  OpBuilder &builder;

  /// Compute circular buffer offset for store (producer side)
  /// Formula: ((global_iter + numStages - 1) % numStages) * stride
  Value computeCircularOffsetStore(Location loc, Value globalIter,
                                    unsigned numStages, int64_t stride);

  /// Compute circular buffer offset for load (consumer side)
  /// Formula: (global_iter % numStages) * stride
  Value computeCircularOffsetLoad(Location loc, Value globalIter,
                                   unsigned numStages, int64_t stride);

  /// Compute global iteration number from potentially nested loops
  Value computeGlobalIteration(scf::ForOp loop);

  /// Decompose pointer into base and indices
  std::pair<Value, SmallVector<Value>> decomposePointer(Value ptr);

  /// Build new pointer with circular buffer dimension
  Value buildPointer(Value baseBuffer, ArrayRef<Value> indices);

  /// Apply swizzle pattern to reduce bank conflicts
  Value applySwizzle(Value ptr, CircularBufferInfo &info);

  /// Substitute loop variable with new value in an operation tree
  void substituteLoopVariable(Operation *op, Value oldVar, Value newVar);
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_CIRCULARBUFFERTRANSFORM_H
