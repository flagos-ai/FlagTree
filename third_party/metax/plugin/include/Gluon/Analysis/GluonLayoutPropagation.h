#ifndef TRITON_METAX_GLUON_LAYOUT_PROPAGATION_H
#define TRITON_METAX_GLUON_LAYOUT_PROPAGATION_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace mlir::triton::gpu::metax::gluon {

inline bool isTransparentLayoutCarrierOp(Operation *op) {
  return isa_and_nonnull<RegionBranchOpInterface,
                         RegionBranchTerminatorOpInterface>(op);
}

inline bool isRegisterToSharedSinkOperand(OpOperand &operand) {
  Operation *owner = operand.getOwner();
  if (auto store = dyn_cast<triton::gpu::LocalStoreOp>(owner))
    return operand.getOperandNumber() == 0 && operand.get() == store.getSrc();
  if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(owner))
    return alloc.getSrc() && operand.getOperandNumber() == 0 &&
           operand.get() == alloc.getSrc();
  return false;
}

inline bool isSupportedDotConstraintEncoding(Attribute encoding) {
  return isa_and_nonnull<triton::gpu::DotOperandEncodingAttr>(encoding);
}

inline bool isSupportedTensorConstraintEncoding(Attribute encoding) {
  return encoding &&
         !isa<triton::gluon::AutoEncodingAttr>(encoding);
}

inline bool isFlexibleTensorLayoutProducer(Operation *op) {
  return isa_and_nonnull<triton::MakeRangeOp, triton::gpu::LocalLoadOp,
                         triton::gpu::BsmPermOp>(op);
}

/// Three-state layout value used by memdesc propagation:
/// uninitialized, one concrete encoding, or unknown/conflicting.
class LayoutEncoding {
public:
  LayoutEncoding() = default;
  LayoutEncoding(Attribute encoding) : encoding(encoding) {}

  bool operator==(const LayoutEncoding &other) const {
    return encoding == other.encoding;
  }
  bool isUninitialized() const { return !encoding.has_value(); }
  bool isUnknown() const { return encoding == nullptr; }
  Attribute getLayoutEncoding() const {
    assert(!isUninitialized() && !isUnknown());
    return *encoding;
  }

  static LayoutEncoding join(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  static LayoutEncoding meet(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  static LayoutEncoding getUnknownLayout() {
    return LayoutEncoding(nullptr);
  }

  void print(raw_ostream &os) const;

private:
  std::optional<Attribute> encoding;
};

inline raw_ostream &operator<<(raw_ostream &os, const LayoutEncoding &layout) {
  layout.print(os);
  return os;
}

class LayoutEncodingLattice
    : public dataflow::Lattice<LayoutEncoding> {
public:
  using Lattice::Lattice;
};

/// Backward propagation of concrete memdesc requirements through the small
/// view whitelist. Shared-memory producers and bridge operations are
/// boundaries; this analysis never chooses a shared layout.
class LayoutBackwardPropagation
    : public dataflow::SparseBackwardDataFlowAnalysis<
          LayoutEncodingLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
                 ArrayRef<const LayoutEncodingLattice *> results) override;
  void visitBranchOperand(OpOperand &operand) override;
  void visitCallOperand(OpOperand &operand) override;
  void setToExitState(LayoutEncodingLattice *lattice) override;

private:
  LogicalResult visitRegionInReverse(Operation *op);
};

/// Forward half of the same memdesc view analysis.
class LayoutForwardPropagation
    : public dataflow::SparseForwardDataFlowAnalysis<
          LayoutEncodingLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const LayoutEncodingLattice *> operands,
                 ArrayRef<LayoutEncodingLattice *> results) override;
  void setToEntryState(LayoutEncodingLattice *lattice) override;

private:
  LogicalResult visitRegion(Operation *op);
};

/// Three-state required register layout. Concrete values originate only from
/// explicit Gluon contracts; unsupported users and conflicts widen to unknown.
class TensorLayout {
public:
  TensorLayout() = default;
  TensorLayout(Attribute encoding) : encoding(encoding) {}

  bool operator==(const TensorLayout &other) const {
    return encoding == other.encoding;
  }
  bool isUninitialized() const { return !encoding.has_value(); }
  bool isUnknown() const { return encoding == nullptr; }
  Attribute getLayoutEncoding() const {
    assert(!isUninitialized() && !isUnknown());
    return *encoding;
  }

  static TensorLayout join(const TensorLayout &lhs,
                           const TensorLayout &rhs);
  static TensorLayout meet(const TensorLayout &lhs,
                           const TensorLayout &rhs);
  static TensorLayout getUnknownLayout() { return TensorLayout(nullptr); }

  void print(raw_ostream &os) const;

private:
  std::optional<Attribute> encoding;
};

inline raw_ostream &operator<<(raw_ostream &os, const TensorLayout &layout) {
  layout.print(os);
  return os;
}

class TensorLayoutLattice : public dataflow::Lattice<TensorLayout> {
public:
  using Lattice::Lattice;
};

/// Close ODS/inference and RegionBranch tensor relations in the already
/// initialized solver. This is pure analysis: it updates only lattice state.
LogicalResult closeTensorLayoutRelations(triton::FuncOp func,
                                         DataFlowSolver &solver);

/// Return values whose RegionBranch predecessors have no single concrete
/// tensor type. The apply pass must keep these carriers as conversion
/// boundaries instead of partially retagging the region.
llvm::DenseSet<Value>
computeBlockedTensorLayoutCarriers(triton::FuncOp func,
                                   DataFlowSolver &solver);

/// Query the concrete type shared by all values after propagation. Returns
/// std::nullopt for an empty set, a conflict, or a blocked carrier.
std::optional<Type> getTensorLayoutConsensusType(
    ArrayRef<Value> values, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues);

/// Backward propagation from explicit set_auto and supported dot-operand
/// requirements. The analysis is read-only and deliberately has no target
/// candidate or legalization knowledge.
class TensorBackwardPropagation
    : public dataflow::SparseBackwardDataFlowAnalysis<TensorLayoutLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<TensorLayoutLattice *> operands,
                 ArrayRef<const TensorLayoutLattice *> results) override;
  void visitBranchOperand(OpOperand &operand) override;
  void visitCallOperand(OpOperand &operand) override;
  void setToExitState(TensorLayoutLattice *lattice) override;
};

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_LAYOUT_PROPAGATION_H
