#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_TMASUPPORT_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_TMASUPPORT_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineOpportunityDetector.h"
#include "triton/Dialect/TritonGPU/Transforms/CircularBufferTransform.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

/// TMA transfer mode
enum class TMAMode {
  Load,       // Global → Shared
  Store,      // Shared → Global
  Multicast   // Global → Shared with multicast
};

/// TMA descriptor configuration
struct TMADescriptor {
  /// Base pointer to global memory
  Value globalPtr;

  /// Destination in shared memory
  Value sharedMemPtr;

  /// Tensor dimensions
  SmallVector<int64_t> shape;

  /// Tensor strides
  SmallVector<int64_t> strides;

  /// Element type
  Type elementType;

  /// Box dimensions for TMA transfer
  SmallVector<int64_t> boxDim;

  /// Transfer mode
  TMAMode mode = TMAMode::Load;

  /// Use multicast (for distributed loads)
  bool useMulticast = false;

  /// Number of stages for async pipeline
  unsigned numStages = 1;

  TMADescriptor() = default;
};

/// Information about TMA transformation
struct TMAInfo {
  /// The loop being transformed
  scf::ForOp loop;

  /// TMA descriptors created
  SmallVector<TMADescriptor> descriptors;

  /// MBarrier for synchronization
  Value mbarrier;

  /// Expected bytes to arrive
  Value expectedBytes;

  /// Phase for multi-stage pipeline
  Value phase;

  /// Pipeline ID
  unsigned pipelineId = 0;

  TMAInfo() = default;
};

/// TMA Support for Hopper GPUs (SM90+)
///
/// This class implements TMA (Tensor Memory Accelerator) support which
/// provides hardware-accelerated bulk data transfers with:
/// - Asynchronous transfers (cp.async.bulk)
/// - Multicast capability for efficient broadcasting
/// - MBarrier synchronization
/// - Hardware-managed transfer completion tracking
///
class TMASupport {
public:
  explicit TMASupport(OpBuilder &builder) : builder(builder) {}

  /// Check if TMA is available on the target hardware
  bool isTMAAvailable() const;

  /// Check if TMA is beneficial for the given opportunity
  bool isProfitable(const PipelineOpportunity &opp,
                    const CircularBufferInfo &circularInfo);

  /// Create TMA descriptor for a tensor transfer
  TMADescriptor createDescriptor(Value globalPtr, Value sharedMemPtr,
                                  ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> strides,
                                  Type elementType);

  /// Apply TMA transformation to replace regular loads
  TMAInfo apply(const PipelineOpportunity &opp,
                CircularBufferInfo &circularInfo,
                unsigned pipelineId);

  /// Insert TMA prefetch for pipeline prologue
  void insertPrefetch(TMAInfo &info, unsigned stageIndex);

  /// Insert TMA wait for pipeline synchronization
  void insertWait(TMAInfo &info);

  /// Create MBarrier for TMA synchronization
  Value createMBarrier(Location loc, unsigned arrivals);

  /// Arrive at MBarrier (producer side)
  void arriveAtMBarrier(Location loc, Value mbarrier, Value bytes);

  /// Wait on MBarrier (consumer side)
  void waitOnMBarrier(Location loc, Value mbarrier, Value phase);

private:
  OpBuilder &builder;

  /// Create cp.async.bulk load operation
  void createAsyncBulkLoad(Location loc, const TMADescriptor &desc,
                           Value mbarrier);

  /// Create cp.async.bulk store operation
  void createAsyncBulkStore(Location loc, const TMADescriptor &desc);

  /// Calculate expected bytes for transfer
  Value calculateExpectedBytes(Location loc, const TMADescriptor &desc);

  /// Transform regular LoadOp to TMA
  void transformLoadToTMA(triton::LoadOp loadOp, TMAInfo &info);

  /// Transform regular StoreOp to TMA
  void transformStoreToTMA(triton::StoreOp storeOp, TMAInfo &info);

  /// Check if operation can use TMA
  bool canUseTMA(Operation *op);

  /// Get compute capability from target
  unsigned getComputeCapability() const;
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TMASUPPORT_H
