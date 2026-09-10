#ifndef MUSATLE_TRANSFORMS_PIPEANALYSIS_H
#define MUSATLE_TRANSFORMS_PIPEANALYSIS_H

#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "MUSATLE/Transforms/PipePartitionUtils.h"
#include "MUSATLE/Transforms/PipeRegionUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::triton::musa_tle {

namespace ttg = ::mlir::triton::gpu;

enum class PipeExecutionMode {
  Unset,
  NonWarpSpecialized,
  StaticWarpSpecialized,
};

enum class PipeLifecycleMode {
  Cyclic,
  OneShot,
};

enum class PipeTransportKind {
  Unknown,
  TME,
  LocalStore,
  // Payload written by ttg.async_copy_global_to_local (llvm.musa.memcpy.g2s).
  // The completion edge is the per-thread async wait issued after the copy;
  // publication uses the same per-warp arrival as LocalStore.
  AsyncCopy,
};

enum class PipeBarrierStorageOwner {
  Pipe,
  External,
};

enum class PipeEndpointRole {
  Writer,
  Reader,
};

enum class PipeBarrierInitialState {
  Pending,
  Ready,
};

enum class PipeReaderDrainKind {
  TMEStore,
};

enum class PipeReaderSubscriptionKind { AllFields, ExplicitSubset };

struct PipeState;

struct PipeCoveredRegion {
  unsigned fieldIndex = 0;
  Value memdescRoot;
  std::optional<int64_t> byteOffset;
  std::optional<int64_t> byteSize;
  bool exact = false;
};

struct PipeCompletionSource {
  PipeTransportKind kind = PipeTransportKind::Unknown;
  Operation *operation = nullptr;
  unsigned destinationField = 0;
  Value stage;
  PipeCoveredRegion coveredRegion;
  int32_t transactionBytes = 0;
  PipeBarrierStorageOwner barrierStorageOwner = PipeBarrierStorageOwner::Pipe;
  // Canonical BarrierAllocOp base for an externally supplied completion
  // barrier.  It is null for pipe-owned completion storage.
  Value externalBarrierRoot;
};

struct PipeCommitGroup {
  Value stage;
  SmallVector<PipeCompletionSource> completionSources;
  int32_t totalTransactionBytes = 0;
  int32_t tmeGroupArrivalCount = 0;
  int32_t localStoreArrivalCount = 0;
  int32_t fullArrivalCount = 0;
  PipeWriterCommitOp commit;
  Value externalBarrierRoot;
};

struct PipeReaderDrainSource {
  PipeReaderDrainKind kind = PipeReaderDrainKind::TMEStore;
  Operation *operation = nullptr;
  unsigned sourceField = 0;
  PipeCoveredRegion coveredRegion;
  Value destinationDescriptor;
};

struct PipeReaderDrainGroup {
  unsigned readerEndpoint = 0;
  Value stage;
  Value phase;
  PipeReaderWaitOp wait;
  SmallVector<PipeReaderDrainSource> drainSources;
  // Unknown aliasing conservatively requires partition-local publication.
  bool sourceModifiedAfterWait = false;
};

struct PipeFieldState {
  unsigned index = 0;
  std::string name;
  Value memdesc;
  Value memdescRoot;
  Type memdescType;
  SmallVector<unsigned> subscribedReaders;
};

struct PipeEndpointState {
  unsigned index = 0;
  std::string name;
  PipeEndpointRole role = PipeEndpointRole::Reader;
  std::optional<PipeReaderSubscriptionKind> readerSubscription;
  SmallVector<unsigned> subscribedFields;
  ttg::WarpSpecializeOp warpSpecialize;
  unsigned partitionIndex = 0;
  PipePartitionKind partition = PipePartitionKind::CTA;
  std::optional<unsigned> worker;
  int32_t warpBegin = 0;
  int32_t warpCount = 0;
};

struct PipePartitionMapping {
  unsigned endpoint = 0;
  ttg::WarpSpecializeOp warpSpecialize;
  unsigned partitionIndex = 0;
  PipePartitionKind partition = PipePartitionKind::CTA;
  std::optional<unsigned> worker;
  int32_t warpBegin = 0;
  int32_t warpCount = 0;
};

struct PipeBarrierRingPlan {
  int32_t capacity = 0;
  int32_t arrivalCount = 0;
  PipeBarrierInitialState initialState = PipeBarrierInitialState::Pending;
  std::optional<int32_t> transactionBytes;
  PipeBarrierStorageOwner storageOwner = PipeBarrierStorageOwner::Pipe;
  Value externalStorage;
};

// The allocation and canonical base are retained in the analysis result so
// LowerPipe can reuse a user-provided full barrier without materializing a
// second pipe-owned ring.
struct PipeExternalBarrierBinding {
  BarrierAllocOp allocation;
  Value base;
  int32_t capacity = 0;
  int32_t arrivalCount = 0;
  PipeBarrierInitialState initialState = PipeBarrierInitialState::Pending;
  std::optional<int32_t> expectBytes;
};

struct PipeCloseTagPlan {
  int32_t capacity = 0;
  bool initialValue = false;
  PipeBarrierStorageOwner storageOwner = PipeBarrierStorageOwner::Pipe;
};

struct PipeCloseGeneration {
  PipeWriterCloseOp close;
  Value stage;
  Value phase;
  int32_t controlArrivalCount = 0;
  int32_t localStoreArrivalCount = 0;
  int32_t fullArrivalCount = 0;
  int32_t transactionBytes = 0;
};

// A logical barrier participant is the stable endpoint/partition contribution
// used by pipe lowering.  Keeping this in the analysis result avoids having
// later warp-specialized lowering re-derive producer/reader warp counts from
// region traversal order.
struct PipeBarrierParticipant {
  unsigned endpointIndex = 0;
  unsigned partitionIndex = 0;
  PipePartitionKind partition = PipePartitionKind::CTA;
  int32_t warpBegin = 0;
  int32_t warpCount = 0;
};

struct PipeBarrierPlan {
  PipeBarrierRingPlan full;
  std::optional<PipeBarrierRingPlan> empty;
  bool hasCloseState = false;
  std::optional<PipeCloseTagPlan> closeTagPlan;
  std::optional<PipeExternalBarrierBinding> externalFull;
  std::optional<PipeBarrierParticipant> writerParticipant;
  SmallVector<PipeBarrierParticipant> readerParticipants;
};

struct PipeLifecycleState {
  PipeLifecycleMode mode = PipeLifecycleMode::Cyclic;
};

struct PipeState {
  PipeCreateOp create;
  int32_t capacity = 0;
  SmallVector<PipeFieldState> fields;
  SmallVector<PipeEndpointState> endpoints;
  SmallVector<std::unique_ptr<PipeCommitGroup>> commitGroups;
  SmallVector<std::unique_ptr<PipeReaderDrainGroup>> readerDrainGroups;
  SmallVector<std::unique_ptr<PipeCloseGeneration>> closeGenerations;
  PipeExecutionMode executionMode = PipeExecutionMode::Unset;
  PipeLifecycleState lifecycle;
  PipeBarrierPlan barrierPlan;
  ttg::WarpSpecializeOp staticWarpSpecialize;

  SmallVector<PipeWriterAcquireOp> acquires;
  SmallVector<PipeWriterCommitOp> commits;
  SmallVector<PipeReaderWaitOp> waits;
  SmallVector<PipeReaderReleaseOp> releases;
  SmallVector<PipeWriterCloseOp> closes;
};

class PipeAnalysisBuilder;

class PipeAnalysisResult {
public:
  PipeState *lookupPipe(Operation *op);
  const PipeState *lookupPipe(Operation *op) const;

  PipeEndpointState *lookupEndpoint(Operation *op);
  const PipeEndpointState *lookupEndpoint(Operation *op) const;

  const PipeCommitGroup *lookupCommitGroup(PipeWriterCommitOp op) const;

  const PipeReaderDrainGroup *lookupReaderDrainGroup(PipeReaderWaitOp op) const;

  const PipeCloseGeneration *lookupCloseGeneration(PipeWriterCloseOp op) const;

  ArrayRef<std::unique_ptr<PipeState>> getPipes() const { return pipes; }
  ArrayRef<Operation *> getLifecycleOps() const { return lifecycleOps; }

private:
  friend class PipeAnalysisBuilder;

  SmallVector<std::unique_ptr<PipeState>> pipes;
  SmallVector<Operation *> lifecycleOps;
  llvm::DenseMap<Operation *, PipeState *> pipeByOperation;
  llvm::DenseMap<Operation *, unsigned> endpointIndexByOperation;
  llvm::DenseMap<Operation *, PipeCommitGroup *> commitGroupByOperation;
  llvm::DenseMap<Operation *, PipeReaderDrainGroup *> readerDrainGroupByWait;
  llvm::DenseMap<Operation *, PipeCloseGeneration *> closeGenerationByOperation;
};

FailureOr<std::unique_ptr<PipeAnalysisResult>>
analyzeMUSAPipes(ModuleOp module);

} // namespace mlir::triton::musa_tle

#endif // __TLE__

#endif // MUSATLE_TRANSFORMS_PIPEANALYSIS_H
