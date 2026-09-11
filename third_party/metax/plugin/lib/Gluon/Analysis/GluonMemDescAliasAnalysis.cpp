#include "Gluon/Analysis/GluonMemDescAliasAnalysis.h"
#include "Gluon/GluonLayoutPlaceholders.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "metax-gluon-memdesc-alias"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace {

bool isMemDesc(Value value) { return isa<ttg::MemDescType>(value.getType()); }

bool isAutoEncoding(Attribute encoding) {
  return isa_and_nonnull<gluon_dialect::AutoEncodingAttr>(
      unwrapNoVerifyEncoding(encoding));
}

SmallVector<int32_t> invertPermutation(ArrayRef<int32_t> order) {
  SmallVector<int32_t> inverse(order.size());
  for (auto [index, dim] : llvm::enumerate(order))
    inverse[dim] = index;
  return inverse;
}

FailureOr<Attribute> inferTransEncoding(Attribute encoding,
                                        ArrayRef<int64_t> shape,
                                        ArrayRef<int32_t> order,
                                        Operation *diagnosticOp) {
  if (!encoding)
    return failure();
  Location location = diagnosticOp ? diagnosticOp->getLoc()
                                    : UnknownLoc::get(encoding.getContext());
  auto *interface =
      encoding.getDialect()
          .getRegisteredInterface<tt::DialectInferLayoutInterface>();
  if (!interface) {
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "encoding dialect does not implement transpose inference: "
          << encoding;
    return failure();
  }

  Attribute result;
  if (failed(interface->inferTransOpEncoding(encoding, shape, order, result,
                                             location))) {
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "failed to project memdesc transpose encoding " << encoding;
    return failure();
  }
  return result;
}

ttg::MemDescType cloneWithEncoding(ttg::MemDescType type,
                                   Attribute encoding) {
  return ttg::MemDescType::get(
      type.getShape(), type.getElementType(), encoding, type.getMemorySpace(),
      type.getMutableMemory(), type.getAllocShape());
}

FailureOr<Attribute> projectReshapeForward(ttg::MemDescReshapeOp reshape,
                                           Attribute sourceEncoding,
                                           Operation *diagnosticOp) {
  auto sourceType =
      cloneWithEncoding(reshape.getSrc().getType(), sourceEncoding);
  auto resultType = cast<ttg::MemDescType>(reshape.getType());
  ttg::MemDescType inferredResultType;
  if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
          reshape.getContext(), reshape.getLoc(), sourceType,
          resultType.getShape(), inferredResultType))) {
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "failed to project memdesc reshape encoding from root";
    return failure();
  }
  return inferredResultType.getEncoding();
}

FailureOr<Attribute> projectReshapeBackward(ttg::MemDescReshapeOp reshape,
                                            Attribute resultEncoding,
                                            Operation *diagnosticOp) {
  auto resultType = cloneWithEncoding(reshape.getType(), resultEncoding);
  auto sourceType = cast<ttg::MemDescType>(reshape.getSrc().getType());
  ttg::MemDescType inferredSourceType;
  if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
          reshape.getContext(), reshape.getLoc(), resultType,
          sourceType.getShape(), inferredSourceType))) {
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "failed to project memdesc reshape encoding to root";
    return failure();
  }
  return inferredSourceType.getEncoding();
}

bool isIdentityEncodingView(Operation *op) {
  if (!isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(op))
    return false;
  auto sourceType = dyn_cast<ttg::MemDescType>(op->getOperand(0).getType());
  auto resultType = dyn_cast<ttg::MemDescType>(op->getResult(0).getType());
  return sourceType && resultType &&
         unwrapNoVerifyEncoding(sourceType.getEncoding()) ==
             unwrapNoVerifyEncoding(resultType.getEncoding());
}

bool haveEquivalentTransform(Operation *lhs, Operation *rhs) {
  if (lhs->getName() != rhs->getName())
    return false;
  if (auto lhsTrans = dyn_cast<ttg::MemDescTransOp>(lhs)) {
    auto rhsTrans = cast<ttg::MemDescTransOp>(rhs);
    auto lhsSource = cast<ttg::MemDescType>(lhsTrans.getSrc().getType());
    auto rhsSource = cast<ttg::MemDescType>(rhsTrans.getSrc().getType());
    auto lhsResult = cast<ttg::MemDescType>(lhsTrans.getType());
    auto rhsResult = cast<ttg::MemDescType>(rhsTrans.getType());
    return lhsTrans.getOrder() == rhsTrans.getOrder() &&
           lhsSource.getShape() == rhsSource.getShape() &&
           lhsResult.getShape() == rhsResult.getShape();
  }
  if (auto lhsReshape = dyn_cast<ttg::MemDescReshapeOp>(lhs)) {
    auto rhsReshape = cast<ttg::MemDescReshapeOp>(rhs);
    auto lhsSource = cast<ttg::MemDescType>(lhsReshape.getSrc().getType());
    auto rhsSource = cast<ttg::MemDescType>(rhsReshape.getSrc().getType());
    auto lhsResult = cast<ttg::MemDescType>(lhsReshape.getType());
    auto rhsResult = cast<ttg::MemDescType>(rhsReshape.getType());
    return lhsSource.getShape() == rhsSource.getShape() &&
           lhsResult.getShape() == rhsResult.getShape();
  }
  if (auto lhsReinterpret = dyn_cast<ttg::MemDescReinterpretOp>(lhs)) {
    auto rhsReinterpret = cast<ttg::MemDescReinterpretOp>(rhs);
    auto lhsSource =
        cast<ttg::MemDescType>(lhsReinterpret.getSrc().getType());
    auto rhsSource =
        cast<ttg::MemDescType>(rhsReinterpret.getSrc().getType());
    auto lhsResult = cast<ttg::MemDescType>(lhsReinterpret.getType());
    auto rhsResult = cast<ttg::MemDescType>(rhsReinterpret.getType());
    return lhsSource.getShape() == rhsSource.getShape() &&
           lhsSource.getElementType() == rhsSource.getElementType() &&
           lhsResult.getShape() == rhsResult.getShape() &&
           lhsResult.getElementType() == rhsResult.getElementType();
  }
  return lhs == rhs;
}

bool haveEquivalentCoordinatePaths(const MemDescAliasPath &lhs,
                                   const MemDescAliasPath &rhs) {
  if (lhs.root != rhs.root)
    return false;

  auto lhsViews = llvm::make_filter_range(
      lhs.views, [](Operation *op) { return !isIdentityEncodingView(op); });
  auto rhsViews = llvm::make_filter_range(
      rhs.views, [](Operation *op) { return !isIdentityEncodingView(op); });
  return llvm::equal(lhsViews, rhsViews, haveEquivalentTransform);
}

} // namespace

Value getMemDescLayoutViewSource(Operation *op) {
  if (!op)
    return {};
  Value source;
  TypeSwitch<Operation *>(op)
      .Case<ttg::MemDescIndexOp>(
          [&](ttg::MemDescIndexOp view) { source = view.getSrc(); })
      .Case<ttg::MemDescSubsliceOp>(
          [&](ttg::MemDescSubsliceOp view) { source = view.getSrc(); })
      .Case<ttg::MemDescTransOp>(
          [&](ttg::MemDescTransOp view) { source = view.getSrc(); })
      .Case<ttg::MemDescReshapeOp>(
          [&](ttg::MemDescReshapeOp view) { source = view.getSrc(); })
      .Case<ttg::MemDescReinterpretOp>(
          [&](ttg::MemDescReinterpretOp view) { source = view.getSrc(); });
  return source;
}

std::optional<unsigned> projectDimensionToRoot(
    const MemDescAliasPath &path, unsigned viewDimension) {
  unsigned dimension = viewDimension;
  for (Operation *view : llvm::reverse(path.views)) {
    Value source = getMemDescLayoutViewSource(view);
    auto sourceType =
        source ? dyn_cast<ttg::MemDescType>(source.getType()) : nullptr;
    auto resultType = view->getNumResults() == 1
                          ? dyn_cast<ttg::MemDescType>(
                                view->getResult(0).getType())
                          : nullptr;
    if (!sourceType || !resultType || dimension >= resultType.getRank())
      return std::nullopt;
    if (isa<ttg::MemDescIndexOp>(view)) {
      ++dimension;
      continue;
    }
    if (auto trans = dyn_cast<ttg::MemDescTransOp>(view)) {
      if (dimension >= trans.getOrder().size() ||
          trans.getOrder()[dimension] < 0)
        return std::nullopt;
      dimension = static_cast<unsigned>(trans.getOrder()[dimension]);
      continue;
    }
    if (isa<ttg::MemDescReshapeOp>(view) &&
        sourceType.getShape() != resultType.getShape())
      return std::nullopt;
    if (isa<ttg::MemDescReinterpretOp>(view) &&
        (sourceType.getShape() != resultType.getShape() ||
         sourceType.getElementType() != resultType.getElementType()))
      return std::nullopt;
    if (sourceType.getRank() != resultType.getRank())
      return std::nullopt;
  }
  return dimension;
}

std::optional<unsigned> projectDimensionFromRoot(
    const MemDescAliasPath &path, unsigned rootDimension) {
  unsigned dimension = rootDimension;
  for (Operation *view : path.views) {
    Value source = getMemDescLayoutViewSource(view);
    auto sourceType =
        source ? dyn_cast<ttg::MemDescType>(source.getType()) : nullptr;
    auto resultType = view->getNumResults() == 1
                          ? dyn_cast<ttg::MemDescType>(
                                view->getResult(0).getType())
                          : nullptr;
    if (!sourceType || !resultType || dimension >= sourceType.getRank())
      return std::nullopt;
    if (isa<ttg::MemDescIndexOp>(view)) {
      if (dimension == 0)
        return std::nullopt;
      --dimension;
      continue;
    }
    if (auto trans = dyn_cast<ttg::MemDescTransOp>(view)) {
      auto found = llvm::find(trans.getOrder(),
                              static_cast<int32_t>(dimension));
      if (found == trans.getOrder().end())
        return std::nullopt;
      dimension =
          static_cast<unsigned>(found - trans.getOrder().begin());
      continue;
    }
    if (isa<ttg::MemDescReshapeOp>(view) &&
        sourceType.getShape() != resultType.getShape())
      return std::nullopt;
    if (isa<ttg::MemDescReinterpretOp>(view) &&
        (sourceType.getShape() != resultType.getShape() ||
         sourceType.getElementType() != resultType.getElementType()))
      return std::nullopt;
    if (sourceType.getRank() != resultType.getRank())
      return std::nullopt;
  }
  return dimension;
}

FailureOr<Attribute> projectEncodingToRoot(const MemDescAliasPath &path,
                                           Attribute viewEncoding,
                                           Operation *diagnosticOp) {
  if (!path.root || !path.value || !viewEncoding)
    return failure();

  Attribute encoding = viewEncoding;
  for (Operation *view : llvm::reverse(path.views)) {
    if (isIdentityEncodingView(view))
      continue;
    if (auto trans = dyn_cast<ttg::MemDescTransOp>(view)) {
      auto resultType = cast<ttg::MemDescType>(trans.getType());
      auto inverseOrder = invertPermutation(trans.getOrder());
      FailureOr<Attribute> projected =
          inferTransEncoding(encoding, resultType.getShape(), inverseOrder,
                             diagnosticOp);
      if (failed(projected))
        return failure();
      encoding = *projected;
      continue;
    }
    if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(view)) {
      FailureOr<Attribute> projected =
          projectReshapeBackward(reshape, encoding, diagnosticOp);
      if (failed(projected))
        return failure();
      encoding = *projected;
      continue;
    }
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "unsupported memdesc view in root encoding projection: "
          << view->getName();
    return failure();
  }
  return encoding;
}

FailureOr<Attribute> projectEncodingFromRoot(const MemDescAliasPath &path,
                                             Attribute rootEncoding,
                                             Operation *diagnosticOp) {
  if (!path.root || !path.value || !rootEncoding)
    return failure();

  Attribute encoding = rootEncoding;
  for (Operation *view : path.views) {
    if (isIdentityEncodingView(view))
      continue;
    if (auto trans = dyn_cast<ttg::MemDescTransOp>(view)) {
      auto sourceType = cast<ttg::MemDescType>(trans.getSrc().getType());
      FailureOr<Attribute> projected =
          inferTransEncoding(encoding, sourceType.getShape(), trans.getOrder(),
                             diagnosticOp);
      if (failed(projected))
        return failure();
      encoding = *projected;
      continue;
    }
    if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(view)) {
      FailureOr<Attribute> projected =
          projectReshapeForward(reshape, encoding, diagnosticOp);
      if (failed(projected))
        return failure();
      encoding = *projected;
      continue;
    }
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "unsupported memdesc view in forward encoding projection: "
          << view->getName();
    return failure();
  }
  return encoding;
}

GluonMemDescAliasAnalysis::State
GluonMemDescAliasAnalysis::transfer(Value value) const {
  if (!isMemDesc(value))
    return {MemDescAliasKnowledge::Unknown, {}};

  // A top-level function argument is an alias root even when the generic
  // RegionBranch analysis observes uses that make it look like a carrier.
  // Region block arguments remain carriers and are joined from their incoming
  // values below.
  if (auto argument = dyn_cast<BlockArgument>(value);
      argument &&
      isa_and_nonnull<tt::FuncOp>(argument.getOwner()->getParentOp()))
    return {MemDescAliasKnowledge::Known, {value, value, {}}};

  // Only SSA values with incoming RegionBranch edges are carrier joins. An
  // init or yielded producer also has an outgoing carrier edge, but it must
  // still be analyzed from its own defining operation (for example, a
  // memdesc_index that selects the next stage of a ring buffer).
  ArrayRef<Value> predecessors = carriers.getPredecessors(value);
  if (!predecessors.empty()) {
    State result;
    for (Value predecessor : predecessors) {
      const State &incoming = lookup(predecessor);
      if (incoming.knowledge == MemDescAliasKnowledge::Bottom)
        continue;
      if (incoming.knowledge == MemDescAliasKnowledge::Unknown)
        return {MemDescAliasKnowledge::Unknown, {}};
      if (result.knowledge == MemDescAliasKnowledge::Bottom) {
        result = incoming;
        result.path.value = value;
        continue;
      }
      if (!haveEquivalentCoordinatePaths(result.path, incoming.path))
        return {MemDescAliasKnowledge::Unknown, {}};
    }
    return result;
  }

  if (isa<BlockArgument>(value))
    return {MemDescAliasKnowledge::Unknown, {}};

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {MemDescAliasKnowledge::Unknown, {}};
  Operation *def = result.getOwner();
  if (isa<ttg::LocalAllocOp, gluon_dialect::RequireLayoutOp,
          gluon_dialect::ReleaseLayoutOp>(def))
    return {MemDescAliasKnowledge::Known, {value, value, {}}};

  if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(def)) {
    auto resultType = dyn_cast<ttg::MemDescType>(reinterpret.getType());
    if (!resultType)
      return {MemDescAliasKnowledge::Unknown, {}};

    Attribute resultEncoding =
        unwrapNoVerifyEncoding(resultType.getEncoding());
    // Reinterpret is an explicit zero-copy family boundary. Its result never
    // propagates layout requirements back to the source. The frontend marker
    // decides whether this independent family is compiler-managed; without the
    // marker its concrete result encoding remains a fixed user contract.
    if (!isAutoEncoding(resultEncoding))
      return {MemDescAliasKnowledge::Known, {value, value, {}}};
    return {MemDescAliasKnowledge::Unknown, {}};
  }

  Value source = getMemDescLayoutViewSource(def);
  if (!source)
    return {MemDescAliasKnowledge::Unknown, {}};
  const State &sourceState = lookup(source);
  if (sourceState.knowledge != MemDescAliasKnowledge::Known)
    return sourceState;

  State projected = sourceState;
  projected.path.value = value;
  projected.path.views.push_back(def);
  return projected;
}

LogicalResult GluonMemDescAliasAnalysis::initialize() {
  if (failed(carriers.initialize()))
    return failure();
  states.clear();

  llvm::SetVector<Value> values;
  root->walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (isMemDesc(result))
        values.insert(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (isMemDesc(argument))
            values.insert(argument);
  });
  for (Value value : values)
    states.try_emplace(value, State{});

  const unsigned maxIterations = std::max<unsigned>(1, values.size() * 3);
  auto runFixedPoint = [&]() {
    bool changed = true;
    unsigned iteration = 0;
    while (changed && iteration++ < maxIterations) {
      changed = false;
      for (Value value : values) {
        State candidate = transfer(value);
        State &current = states[value];
        if (candidate.knowledge == MemDescAliasKnowledge::Bottom ||
            current.knowledge == MemDescAliasKnowledge::Unknown)
          continue;
        if (candidate.knowledge == MemDescAliasKnowledge::Unknown) {
          current = std::move(candidate);
          changed = true;
          continue;
        }
        if (current.knowledge == MemDescAliasKnowledge::Bottom) {
          current = std::move(candidate);
          changed = true;
          continue;
        }
        if (!haveEquivalentCoordinatePaths(current.path, candidate.path)) {
          current = {MemDescAliasKnowledge::Unknown, {}};
          changed = true;
        }
      }
    }
    return !changed;
  };

  if (!runFixedPoint())
    return root->emitError()
           << "logical memdesc alias analysis did not reach a bounded "
              "fixed point after "
           << maxIterations << " iterations";

  // A closed carrier cycle with no allocation or function-entry seed has no
  // provable family. Promote unresolved bottom states and propagate that fact.
  for (Value value : values)
    if (states[value].knowledge == MemDescAliasKnowledge::Bottom)
      states[value] = {MemDescAliasKnowledge::Unknown, {}};
  if (!runFixedPoint())
    return root->emitError()
           << "logical memdesc alias analysis did not stabilize after "
              "resolving unseeded carrier cycles";

  for (Value value : values) {
    const State &state = states.lookup(value);
    LDBG("[memdesc-alias] value=" << value << " state="
                                  << static_cast<unsigned>(state.knowledge)
                                  << " root=" << state.path.root
                                  << " views=" << state.path.views.size());
  }
  return success();
}

const GluonMemDescAliasAnalysis::State &
GluonMemDescAliasAnalysis::lookup(Value value) const {
  auto it = states.find(value);
  return it == states.end() ? unknownState : it->second;
}

FailureOr<MemDescAliasPath>
GluonMemDescAliasAnalysis::getPath(Value value) const {
  const State &state = lookup(value);
  if (state.knowledge != MemDescAliasKnowledge::Known)
    return failure();
  return state.path;
}

FailureOr<LayoutFamilyId>
GluonMemDescAliasAnalysis::getFamily(Value value) const {
  FailureOr<MemDescAliasPath> path = getPath(value);
  if (failed(path))
    return failure();
  return LayoutFamilyId{path->root};
}

} // namespace mlir::triton::gpu::metax::gluon
