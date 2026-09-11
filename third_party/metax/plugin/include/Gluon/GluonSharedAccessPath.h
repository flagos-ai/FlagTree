#ifndef TRITON_METAX_GLUON_SHARED_ACCESS_PATH_H
#define TRITON_METAX_GLUON_SHARED_ACCESS_PATH_H

#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace mlir::triton::gpu::metax::gluon {

enum class SharedAccessCoverage { Whole, ExactSlot, Partial, Unknown };

/// One root-coordinate index in a shared access path. Constants compare by
/// value, while dynamic indices compare by SSA identity; two unrelated dynamic
/// selectors must never collapse to the same logical slot.
struct SharedAccessPathIndex {
  std::optional<int64_t> constant;
  Value dynamic;

  bool operator==(const SharedAccessPathIndex &other) const {
    return constant == other.constant && dynamic == other.dynamic;
  }
};

struct SharedAccessPathFact {
  Value logicalRoot;
  SharedAccessCoverage coverage = SharedAccessCoverage::Unknown;
  std::optional<int64_t> slot;
  SmallVector<SharedAccessPathIndex, 4> indices;
  SmallVector<int64_t, 4> offsets;
  /// Static byte base relative to `logicalRoot` and the logical offsets carried
  /// by the lowered SharedMemoryObject. These are analysis facts, not a claim
  /// that the logical view is physically contiguous.
  std::optional<uint64_t> byteBase;
  SmallVector<int64_t, 4> logicalOffsets;
  /// Conservative root-relative byte interval touched by this exact view. A
  /// missing interval means that physical confinement could not be proven.
  std::optional<std::pair<uint64_t, uint64_t>> byteInterval;
};

struct SharedAccessPathState {
  SmallVector<SharedAccessPathFact, 2> alternatives;
  bool unknown = false;

  bool operator==(const SharedAccessPathState &other) const;
};

/// A bounded, monotone forward analysis of logical shared-memory access paths.
/// It deliberately does not use late physical allocation identity: definite
/// initialization is a logical value/path property, while physical intervals
/// are overlaid later for may-alias hazards.
class SharedAccessPathAnalysis {
public:
  explicit SharedAccessPathAnalysis(Operation *root)
      : root(root), carriers(root) {}

  LogicalResult initialize();
  const SharedAccessPathState &lookup(Value value) const;

  /// Return a stable exact-access identity. Transparent carriers inherit an
  /// identity only when all incoming paths agree by SSA correlation or by one
  /// identical singleton access fact; mixed-slot joins keep their own identity.
  Value getCarrierIdentity(Value value) const;

  /// Return the access identity whose complete initialization covers `value`.
  /// Coverage-narrowing views keep a distinct exact identity but name their
  /// immediate source here, preserving the direction whole -> partial.
  Value getCoveringIdentity(Value value) const;

  ArrayRef<RegionCarrierEdge> getCarrierEdges() const {
    return carriers.getEdges();
  }

private:
  SharedAccessPathState transfer(Value value) const;
  void buildCarrierIdentities(ArrayRef<Value> values);

  Operation *root;
  GluonRegionBranchAnalysis carriers;
  DenseMap<Value, SharedAccessPathState> states;
  DenseMap<Value, Value> carrierIdentities;
  SharedAccessPathState unknownState;
};

StringRef stringifySharedAccessCoverage(SharedAccessCoverage coverage);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_SHARED_ACCESS_PATH_H
