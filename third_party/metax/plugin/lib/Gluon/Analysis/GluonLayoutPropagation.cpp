#include "Gluon/Analysis/GluonLayoutPropagation.h"

#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "metax-gluon-layout-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace {

FailureOr<Attribute> inferTransEncoding(Attribute encoding,
                                        ArrayRef<int64_t> shape,
                                        ArrayRef<int32_t> order,
                                        Location location) {
  if (!encoding)
    return failure();
  auto *interface =
      encoding.getDialect()
          .getRegisteredInterface<tt::DialectInferLayoutInterface>();
  if (!interface)
    return failure();
  Attribute result;
  if (failed(interface->inferTransOpEncoding(encoding, shape, order, result,
                                             location)))
    return failure();
  return result;
}

SmallVector<int32_t> invertPermutation(ArrayRef<int32_t> order) {
  SmallVector<int32_t> inverse(order.size());
  for (auto [index, dimension] : llvm::enumerate(order))
    inverse[dimension] = index;
  return inverse;
}

bool isAuto(Attribute encoding) {
  return isa_and_nonnull<gd::AutoEncodingAttr>(encoding);
}

bool isTrackedTensor(Value value) {
  return isa<RankedTensorType>(value.getType());
}

bool isIdentityMemDescView(Operation *op) {
  if (!isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(op))
    return false;
  auto source = dyn_cast<ttg::MemDescType>(op->getOperand(0).getType());
  auto result = dyn_cast<ttg::MemDescType>(op->getResult(0).getType());
  return source && result &&
         unwrapNoVerifyEncoding(source.getEncoding()) ==
             unwrapNoVerifyEncoding(result.getEncoding());
}

bool isAllowedTensorUser(Operation *op, unsigned operandIndex) {
  if (auto require = dyn_cast<gd::RequireLayoutOp>(op)) {
    auto type = dyn_cast<RankedTensorType>(require.getType());
    return operandIndex == 0 && type &&
           isSupportedDotConstraintEncoding(type.getEncoding());
  }
  if (isa<gd::SetAutoLayoutOp, ttg::ConvertLayoutOp>(op))
    return operandIndex == 0;
  if (isTransparentLayoutCarrierOp(op) ||
      gd::hasSameTensorEncodingRelation(op) ||
      gd::hasSameLoadStoreTensorEncodingRelation(op))
    return true;
  if (operandIndex < op->getNumOperands() &&
      isRegisterToSharedSinkOperand(op->getOpOperand(operandIndex)))
    return true;
  if (auto dot = dyn_cast<tt::DotOp>(op))
    return operandIndex == 2 && dot.getC() == op->getOperand(operandIndex);
  return false;
}

} // namespace

void LayoutEncoding::print(raw_ostream &os) const {
  if (isUninitialized())
    os << "<UNINITIALIZED>";
  else if (isUnknown())
    os << "<UNKNOWN>";
  else
    getLayoutEncoding().print(os);
}

LayoutEncoding LayoutEncoding::join(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  return lhs == rhs ? lhs : getUnknownLayout();
}

LayoutEncoding LayoutEncoding::meet(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  return join(lhs, rhs);
}

LogicalResult
LayoutBackwardPropagation::visitRegionInReverse(Operation *op) {
  for (Region &region : llvm::reverse(op->getRegions()))
    for (Block &block : llvm::reverse(region))
      for (Operation &nested : llvm::reverse(block)) {
        SmallVector<LayoutEncodingLattice *> operands;
        SmallVector<const LayoutEncodingLattice *> results;
        llvm::transform(nested.getOperands(), std::back_inserter(operands),
                        [&](Value value) { return getLatticeElement(value); });
        llvm::transform(nested.getResults(), std::back_inserter(results),
                        [&](Value value) { return getLatticeElement(value); });
        if (failed(visitOperation(&nested, operands, results)))
          return failure();
      }
  return success();
}

LogicalResult LayoutBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
    ArrayRef<const LayoutEncodingLattice *> results) {
  if (isa<gd::ReleaseLayoutOp>(op))
    return success();
  if (isa<RegionBranchOpInterface>(op))
    return visitRegionInReverse(op);

  if (auto require = dyn_cast<gd::RequireLayoutOp>(op)) {
    auto requiredType = dyn_cast<ttg::MemDescType>(require.getType());
    if (!requiredType)
      return success();
    ChangeResult changed =
        operands.front()->meet(LayoutEncoding(requiredType.getEncoding()));
    propagateIfChanged(operands.front(), changed);
    return success();
  }

  if (isIdentityMemDescView(op)) {
    ChangeResult changed = operands.front()->meet(results.front()->getValue());
    propagateIfChanged(operands.front(), changed);
    return success();
  }

  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op)) {
    const LayoutEncoding &state = results.front()->getValue();
    if (state.isUninitialized() || state.isUnknown())
      return success();
    auto resultType = cast<ttg::MemDescType>(trans.getType());
    FailureOr<Attribute> source = inferTransEncoding(
        state.getLayoutEncoding(), resultType.getShape(),
        invertPermutation(trans.getOrder()), op->getLoc());
    if (failed(source)) {
      ChangeResult changed =
          operands.front()->meet(LayoutEncoding::getUnknownLayout());
      propagateIfChanged(operands.front(), changed);
      return success();
    }
    ChangeResult changed =
        operands.front()->meet(LayoutEncoding(*source));
    propagateIfChanged(operands.front(), changed);
    return success();
  }

  if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
    const LayoutEncoding &state = results.front()->getValue();
    if (state.isUninitialized() || state.isUnknown())
      return success();
    auto resultType = cast<ttg::MemDescType>(reshape.getType());
    auto sourceType = cast<ttg::MemDescType>(reshape.getSrc().getType());
    auto concreteResult = ttg::MemDescType::get(
        resultType.getShape(), resultType.getElementType(),
        state.getLayoutEncoding(), resultType.getMemorySpace(),
        resultType.getMutableMemory(), resultType.getAllocShape());
    ttg::MemDescType inferredSource;
    if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
            op->getContext(), op->getLoc(), concreteResult,
            sourceType.getShape(), inferredSource))) {
      ChangeResult changed =
          operands.front()->meet(LayoutEncoding::getUnknownLayout());
      propagateIfChanged(operands.front(), changed);
      return success();
    }
    ChangeResult changed = operands.front()->meet(
        LayoutEncoding(inferredSource.getEncoding()));
    propagateIfChanged(operands.front(), changed);
    return success();
  }

  // Reinterpret without equal declared encodings has no inference interface.
  // Local/shared/async producers are bridge boundaries. All other memdesc ops
  // are opaque to this analysis.
  return success();
}

void LayoutBackwardPropagation::visitBranchOperand(OpOperand &) {}

void LayoutBackwardPropagation::visitCallOperand(OpOperand &) {
  llvm_unreachable("Gluon layout propagation requires inlined calls");
}

void LayoutBackwardPropagation::setToExitState(LayoutEncodingLattice *) {}

LogicalResult LayoutForwardPropagation::visitRegion(Operation *op) {
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block) {
        SmallVector<const LayoutEncodingLattice *> operands;
        SmallVector<LayoutEncodingLattice *> results;
        llvm::transform(nested.getOperands(), std::back_inserter(operands),
                        [&](Value value) { return getLatticeElement(value); });
        llvm::transform(nested.getResults(), std::back_inserter(results),
                        [&](Value value) { return getLatticeElement(value); });
        if (failed(visitOperation(&nested, operands, results)))
          return failure();
      }
  return success();
}

LogicalResult LayoutForwardPropagation::visitOperation(
    Operation *op, ArrayRef<const LayoutEncodingLattice *> operands,
    ArrayRef<LayoutEncodingLattice *> results) {
  if (isa<RegionBranchOpInterface>(op))
    return visitRegion(op);
  if (operands.empty() || results.empty())
    return success();

  LayoutEncoding state = operands.front()->getValue();
  if (isIdentityMemDescView(op)) {
    ChangeResult changed = results.front()->meet(state);
    propagateIfChanged(results.front(), changed);
    return success();
  }

  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op)) {
    if (!state.isUninitialized() && !state.isUnknown()) {
      auto sourceType = cast<ttg::MemDescType>(trans.getSrc().getType());
      FailureOr<Attribute> result = inferTransEncoding(
          state.getLayoutEncoding(), sourceType.getShape(), trans.getOrder(),
          op->getLoc());
      state = failed(result) ? LayoutEncoding::getUnknownLayout()
                             : LayoutEncoding(*result);
    }
  } else if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
    if (!state.isUninitialized() && !state.isUnknown()) {
      auto sourceType = cast<ttg::MemDescType>(reshape.getSrc().getType());
      auto concreteSource = ttg::MemDescType::get(
          sourceType.getShape(), sourceType.getElementType(),
          state.getLayoutEncoding(), sourceType.getMemorySpace(),
          sourceType.getMutableMemory(), sourceType.getAllocShape());
      auto resultType = cast<ttg::MemDescType>(reshape.getType());
      ttg::MemDescType inferredResult;
      if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
              op->getContext(), op->getLoc(), concreteSource,
              resultType.getShape(), inferredResult)))
        state = LayoutEncoding::getUnknownLayout();
      else
        state = LayoutEncoding(inferredResult.getEncoding());
    }
  } else {
    // Includes non-identity reinterpret and every bridge operation.
    return success();
  }

  ChangeResult changed = results.front()->meet(state);
  propagateIfChanged(results.front(), changed);
  return success();
}

void LayoutForwardPropagation::setToEntryState(LayoutEncodingLattice *) {}

void TensorLayout::print(raw_ostream &os) const {
  if (isUninitialized())
    os << "<UNINITIALIZED>";
  else if (isUnknown())
    os << "<UNKNOWN>";
  else
    getLayoutEncoding().print(os);
}

TensorLayout TensorLayout::join(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  return lhs == rhs ? lhs : getUnknownLayout();
}

TensorLayout TensorLayout::meet(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  return join(lhs, rhs);
}

LogicalResult TensorBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TensorLayoutLattice *> operands,
    ArrayRef<const TensorLayoutLattice *> results) {
  if (auto seed = dyn_cast<gd::SetAutoLayoutOp>(op)) {
    auto type = cast<RankedTensorType>(seed.getType());
    ChangeResult changed =
        operands.front()->meet(TensorLayout(type.getEncoding()));
    propagateIfChanged(operands.front(), changed);
    return success();
  }

  if (auto require = dyn_cast<gd::RequireLayoutOp>(op)) {
    auto type = dyn_cast<RankedTensorType>(require.getType());
    if (!type || !isSupportedDotConstraintEncoding(type.getEncoding()))
      return success();
    ChangeResult changed =
        operands.front()->meet(TensorLayout(type.getEncoding()));
    propagateIfChanged(operands.front(), changed);
    return success();
  }

  if (isa<gd::ReleaseLayoutOp>(op))
    return success();

  const bool exactSame =
      gd::hasSameTensorEncodingRelation(op) ||
      gd::hasSameLoadStoreTensorEncodingRelation(op);
  if (exactSame) {
    TensorLayout consensus;
    for (auto [index, operand] : llvm::enumerate(op->getOperands()))
      if (isTrackedTensor(operand))
        consensus =
            TensorLayout::meet(consensus, operands[index]->getValue());
    for (auto [index, result] : llvm::enumerate(op->getResults()))
      if (isTrackedTensor(result))
        consensus =
            TensorLayout::meet(consensus, results[index]->getValue());

    if (!consensus.isUninitialized()) {
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        if (!isTrackedTensor(operand))
          continue;
        ChangeResult changed = operands[index]->meet(consensus);
        propagateIfChanged(operands[index], changed);
      }
      for (Value result : op->getResults()) {
        if (!isTrackedTensor(result))
          continue;
        TensorLayoutLattice *lattice = getLatticeElement(result);
        ChangeResult changed = lattice->meet(consensus);
        propagateIfChanged(lattice, changed);
      }
    }
  } else if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    const TensorLayout &state = results.front()->getValue();
    if (!state.isUninitialized() && !state.isUnknown()) {
      ChangeResult changed = operands.front()->meet(state);
      propagateIfChanged(operands.front(), changed);
    }
  } else if (auto dot = dyn_cast<tt::DotOp>(op)) {
    const TensorLayout &state = results.front()->getValue();
    if (!state.isUninitialized() && !state.isUnknown()) {
      ChangeResult changed = operands[2]->meet(state);
      propagateIfChanged(operands[2], changed);
    }
  } else if (!results.empty()) {
    TensorLayout consensus;
    for (const TensorLayoutLattice *result : results)
      consensus = TensorLayout::meet(consensus, result->getValue());
    if (!consensus.isUninitialized() && !consensus.isUnknown()) {
      Attribute sourceEncoding =
          ::mlir::inferSrcEncoding(op, consensus.getLayoutEncoding());
      if (sourceEncoding) {
        for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
          if (!isTrackedTensor(operand))
            continue;
          ChangeResult changed =
              operands[index]->meet(TensorLayout(sourceEncoding));
          propagateIfChanged(operands[index], changed);
        }
      }
    }
  }

  if (!exactSame && !isa<ttg::ConvertLayoutOp, tt::DotOp>(op)) {
    TensorLayout operandConsensus;
    for (auto [index, operand] : llvm::enumerate(op->getOperands()))
      if (isTrackedTensor(operand))
        operandConsensus = TensorLayout::meet(
            operandConsensus, operands[index]->getValue());
    if (!operandConsensus.isUninitialized() &&
        !operandConsensus.isUnknown()) {
      Attribute resultEncoding = ::mlir::inferDstEncoding(
          op, operandConsensus.getLayoutEncoding());
      if (resultEncoding)
        for (Value result : op->getResults()) {
          if (!isTrackedTensor(result))
            continue;
          TensorLayoutLattice *lattice = getLatticeElement(result);
          ChangeResult changed =
              lattice->meet(TensorLayout(resultEncoding));
          propagateIfChanged(lattice, changed);
        }
    }
  }

  // A concrete requirement can be absorbed only if every use of the value is
  // an explicit boundary or a declared transparent relation.
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (!isTrackedTensor(operand) || isAllowedTensorUser(op, index))
      continue;
    const TensorLayout &state = operands[index]->getValue();
    if (state.isUninitialized())
      continue;
    ChangeResult changed =
        operands[index]->meet(TensorLayout::getUnknownLayout());
    propagateIfChanged(operands[index], changed);
  }

  // Fixed/opaque producers are conversion boundaries. The apply pass may
  // retag only operations whose layout semantics it can prove.
  bool transparentProducer =
      isFlexibleTensorLayoutProducer(op) ||
      isTransparentLayoutCarrierOp(op) ||
      gd::hasSameTensorEncodingRelation(op) ||
      isa<arith::ConstantOp, tt::DotOp>(op);
  if (!transparentProducer) {
    for (Value result : op->getResults()) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !isAuto(type.getEncoding()))
        continue;
      TensorLayoutLattice *lattice = getLatticeElement(result);
      if (lattice->getValue().isUninitialized())
        continue;
      Attribute desired = lattice->getValue().isUnknown()
                              ? Attribute()
                              : lattice->getValue().getLayoutEncoding();
      if (desired && ::mlir::inferSrcEncoding(op, desired))
        continue;
      ChangeResult changed =
          lattice->meet(TensorLayout::getUnknownLayout());
      propagateIfChanged(lattice, changed);
    }
  }
  return success();
}

void TensorBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  if (!isTrackedTensor(operand.get()) ||
      isTransparentLayoutCarrierOp(operand.getOwner()))
    return;
  TensorLayoutLattice *lattice = getLatticeElement(operand.get());
  if (lattice->getValue().isUninitialized())
    return;
  ChangeResult changed =
      lattice->meet(TensorLayout::getUnknownLayout());
  propagateIfChanged(lattice, changed);
}

void TensorBackwardPropagation::visitCallOperand(OpOperand &operand) {
  if (!isTrackedTensor(operand.get()))
    return;
  TensorLayoutLattice *lattice = getLatticeElement(operand.get());
  if (lattice->getValue().isUninitialized())
    return;
  ChangeResult changed =
      lattice->meet(TensorLayout::getUnknownLayout());
  propagateIfChanged(lattice, changed);
}

void TensorBackwardPropagation::setToExitState(TensorLayoutLattice *) {}

namespace {

bool isAutoTensorValue(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && isAuto(unwrapNoVerifyEncoding(type.getEncoding()));
}

Type getTensorCandidateType(
    Value value, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues) {
  auto type = cast<RankedTensorType>(value.getType());
  if (!isAutoTensorValue(value) || blockedValues.contains(value))
    return type;
  auto *lattice = solver.lookupState<TensorLayoutLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized() ||
      lattice->getValue().isUnknown())
    return type;
  return RankedTensorType::get(
      type.getShape(), type.getElementType(),
      lattice->getValue().getLayoutEncoding());
}

TensorLayout getTensorState(Value value, DataFlowSolver &solver) {
  if (!isa<RankedTensorType>(value.getType()))
    return {};
  auto *state = solver.lookupState<TensorLayoutLattice>(value);
  return state ? state->getValue() : TensorLayout();
}

bool meetTensorState(Value value, const TensorLayout &incoming,
                     DataFlowSolver &solver) {
  if (!isa<RankedTensorType>(value.getType()) ||
      incoming.isUninitialized())
    return false;
  auto *state = solver.getOrCreateState<TensorLayoutLattice>(value);
  return state->meet(incoming) == ChangeResult::Change;
}

TensorLayout meetTensorValues(ValueRange values, DataFlowSolver &solver) {
  TensorLayout consensus;
  for (Value value : values)
    if (isa<RankedTensorType>(value.getType()))
      consensus =
          TensorLayout::meet(consensus, getTensorState(value, solver));
  return consensus;
}

} // namespace

std::optional<Type> getTensorLayoutConsensusType(
    ArrayRef<Value> values, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues) {
  if (values.empty())
    return std::nullopt;
  std::optional<Type> consensus;
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      return std::nullopt;
    Type candidate =
        getTensorCandidateType(value, solver, blockedValues);
    if (!consensus)
      consensus = candidate;
    else if (*consensus != candidate)
      return std::nullopt;
  }
  return consensus;
}

llvm::DenseSet<Value>
computeBlockedTensorLayoutCarriers(tt::FuncOp func,
                                   DataFlowSolver &solver) {
  GluonRegionBranchAnalysis carriers(func);
  (void)carriers.initialize();

  llvm::DenseSet<Value> blocked;
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::DenseSet<Value> seen;
    for (const RegionCarrierEdge &edge : carriers.getEdges()) {
      Value input = edge.successorInput;
      if (!isa<RankedTensorType>(input.getType()) ||
          !seen.insert(input).second)
        continue;
      ArrayRef<Value> predecessors = carriers.getPredecessors(input);
      if (predecessors.empty() ||
          getTensorLayoutConsensusType(predecessors, solver, blocked))
        continue;
      changed |= blocked.insert(input).second;
      for (Value predecessor : predecessors)
        if (isa<RankedTensorType>(predecessor.getType()))
          changed |= blocked.insert(predecessor).second;
    }
  }
  return blocked;
}

LogicalResult closeTensorLayoutRelations(tt::FuncOp func,
                                         DataFlowSolver &solver) {
  SmallVector<Value> tensorValues;
  func.walk([&](Operation *op) {
    llvm::copy_if(op->getResults(), std::back_inserter(tensorValues),
                  [](Value value) {
                    return isa<RankedTensorType>(value.getType());
                  });
    for (Region &region : op->getRegions())
      for (Block &block : region)
        llvm::copy_if(block.getArguments(),
                      std::back_inserter(tensorValues), [](Value value) {
                        return isa<RankedTensorType>(value.getType());
                      });
  });

  // Concrete IR types are hard facts at exact operation boundaries.
  for (Value value : tensorValues) {
    auto type = cast<RankedTensorType>(value.getType());
    Attribute encoding = unwrapNoVerifyEncoding(type.getEncoding());
    if (!isAuto(encoding))
      (void)meetTensorState(value, TensorLayout(encoding), solver);
  }

  GluonRegionBranchAnalysis carriers(func);
  if (failed(carriers.initialize()))
    return failure();

  const unsigned maxIterations =
      std::max<unsigned>(1, tensorValues.size() * 2 + 1);
  bool changed = true;
  unsigned iteration = 0;
  while (changed && iteration++ < maxIterations) {
    changed = false;
    func.walk([&](Operation *op) {
      // Layout contracts and explicit conversions are boundaries, not
      // ordinary ODS exact-same relations. Their dedicated sparse-analysis
      // rules decide which side may propagate; treating them generically here
      // would retag a fixed producer instead of preserving the conversion.
      if (isa<gd::SetAutoLayoutOp, gd::RequireLayoutOp,
              gd::ReleaseLayoutOp, ttg::ConvertLayoutOp>(op))
        return;

      const bool exactSame =
          gd::hasSameTensorEncodingRelation(op) ||
          gd::hasSameLoadStoreTensorEncodingRelation(op);
      if (exactSame) {
        SmallVector<Value> related;
        llvm::copy_if(op->getOperands(), std::back_inserter(related),
                      [](Value value) {
                        return isa<RankedTensorType>(value.getType());
                      });
        llvm::copy_if(op->getResults(), std::back_inserter(related),
                      [](Value value) {
                        return isa<RankedTensorType>(value.getType());
                      });
        TensorLayout consensus = meetTensorValues(related, solver);
        for (Value value : related)
          changed |= meetTensorState(value, consensus, solver);
        return;
      }

      TensorLayout operandConsensus =
          meetTensorValues(op->getOperands(), solver);
      if (!operandConsensus.isUninitialized() &&
          !operandConsensus.isUnknown()) {
        Attribute resultEncoding = ::mlir::inferDstEncoding(
            op, operandConsensus.getLayoutEncoding());
        if (resultEncoding)
          for (Value result : op->getResults())
            if (isa<RankedTensorType>(result.getType()))
              changed |= meetTensorState(
                  result, TensorLayout(resultEncoding), solver);
      }

      TensorLayout resultConsensus =
          meetTensorValues(op->getResults(), solver);
      if (!resultConsensus.isUninitialized() &&
          !resultConsensus.isUnknown()) {
        Attribute sourceEncoding = ::mlir::inferSrcEncoding(
            op, resultConsensus.getLayoutEncoding());
        if (sourceEncoding)
          for (Value operand : op->getOperands())
            if (isa<RankedTensorType>(operand.getType()))
              changed |= meetTensorState(
                  operand, TensorLayout(sourceEncoding), solver);
      }
    });

    llvm::DenseSet<Value> seen;
    for (const RegionCarrierEdge &edge : carriers.getEdges()) {
      Value input = edge.successorInput;
      if (!isa<RankedTensorType>(input.getType()) ||
          !seen.insert(input).second)
        continue;
      SmallVector<Value> related(carriers.getPredecessors(input));
      related.push_back(input);
      TensorLayout consensus = meetTensorValues(related, solver);
      for (Value value : related)
        changed |= meetTensorState(value, consensus, solver);
    }
  }

  if (changed)
    return func.emitError()
           << "Gluon tensor layout relations did not reach a bounded "
              "three-state fixed point";
  return success();
}

} // namespace mlir::triton::gpu::metax::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
