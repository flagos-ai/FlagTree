#ifndef TRITON_METAX_GLUON_REGION_BRANCH_ANALYSIS_H
#define TRITON_METAX_GLUON_REGION_BRANCH_ANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::gpu::metax::gluon {

bool appendRegionCarrierPredecessors(
    Value value, SmallVectorImpl<Value> &predecessors);

/// One transparent SSA carrier edge exposed by RegionBranchOpInterface.
/// Layout and memory analyses consume the same graph instead of independently
/// recognizing scf.for/scf.if/scf.while syntax.
struct RegionCarrierEdge {
  Operation *owner = nullptr;
  /// Null denotes the parent point of `owner`; otherwise this is the source
  /// region whose terminator supplies `predecessor`.
  Region *predecessorRegion = nullptr;
  /// Null denotes the parent point of `owner`; otherwise this is the target
  /// region whose entry argument is `successorInput`.
  Region *successorRegion = nullptr;
  Value predecessor;
  Value successorInput;
};

class GluonRegionBranchAnalysis {
public:
  explicit GluonRegionBranchAnalysis(Operation *root) : root(root) {}

  LogicalResult initialize();

  ArrayRef<RegionCarrierEdge> getEdges() const { return edges; }
  ArrayRef<Value> getPredecessors(Value value) const;
  ArrayRef<Value> getSuccessors(Value value) const;
  bool isCarrier(Value value) const {
    return predecessors.contains(value) || successors.contains(value);
  }

private:
  Operation *root;
  SmallVector<RegionCarrierEdge> edges;
  DenseMap<Value, SmallVector<Value, 2>> predecessors;
  DenseMap<Value, SmallVector<Value, 2>> successors;
};

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_REGION_BRANCH_ANALYSIS_H
