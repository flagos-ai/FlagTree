#include "Gluon/Analysis/GluonLayoutPropagation.h"
#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Analysis/GluonDotAnalysis.h"
#include "Gluon/Analysis/GluonMemDescAliasAnalysis.h"
#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"
#include "Gluon/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "metax-gluon-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;
namespace layout = ::mlir::triton::gpu::metax::gluon;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONPROPAGATELAYOUT
#include "Gluon/Passes.h.inc"

namespace {

bool isAutoEncoding(Attribute encoding) {
  return isa_and_nonnull<gd::AutoEncodingAttr>(
      layout::unwrapNoVerifyEncoding(encoding));
}

bool isAutoTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && isAutoEncoding(type.getEncoding());
}

bool isAutoMemDesc(
    Value value,
    const layout::GluonMemDescAliasAnalysis &aliases) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type)
    return false;
  if (isAutoEncoding(type.getEncoding()))
    return true;
  FailureOr<layout::MemDescAliasPath> path = aliases.getPath(value);
  if (failed(path))
    return false;
  return layout::isCompilerManagedSharedFamilyRoot(path->root);
}

RankedTensorType cloneTensor(RankedTensorType type, Attribute encoding) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encoding);
}

ttg::MemDescType cloneMemDesc(ttg::MemDescType type, Attribute encoding) {
  return ttg::MemDescType::get(
      type.getShape(), type.getElementType(), encoding, type.getMemorySpace(),
      type.getMutableMemory(), type.getAllocShape());
}

/// Layout propagation is structural, so every syntactic region edge must
/// participate even when a condition currently folds to a constant.
class PathInsensitiveConstantPropagation final
    : public dataflow::SparseForwardDataFlowAnalysis<
          dataflow::Lattice<dataflow::ConstantValue>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(
      Operation *,
      ArrayRef<const dataflow::Lattice<dataflow::ConstantValue> *>,
      ArrayRef<dataflow::Lattice<dataflow::ConstantValue> *> results) override {
    for (auto *result : results)
      propagateIfChanged(
          result, result->join(dataflow::ConstantValue::getUnknownConstant()));
    return success();
  }

  void setToEntryState(
      dataflow::Lattice<dataflow::ConstantValue> *lattice) override {
    propagateIfChanged(
        lattice,
        lattice->join(dataflow::ConstantValue::getUnknownConstant()));
  }
};

bool hasExplicitLayoutContract(tt::FuncOp func) {
  WalkResult found = func.walk([&](Operation *op) {
    return isa<gd::SetAutoLayoutOp, gd::RequireLayoutOp,
               gd::ReleaseLayoutOp>(op)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return found.wasInterrupted();
}

void updateTensorCarrierTypes(
    tt::FuncOp func, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues) {
  layout::GluonRegionBranchAnalysis carriers(func);
  (void)carriers.initialize();
  DenseSet<Value> seen;
  for (const layout::RegionCarrierEdge &edge :
       llvm::reverse(carriers.getEdges())) {
    Value input = edge.successorInput;
    if (!isa<RankedTensorType>(input.getType()) ||
        !seen.insert(input).second)
      continue;
    std::optional<Type> consensus = layout::getTensorLayoutConsensusType(
        carriers.getPredecessors(input), solver, blockedValues);
    if (consensus && input.getType() != *consensus)
      input.setType(*consensus);
  }
}

LogicalResult retagConstant(arith::ConstantOp constant,
                            RankedTensorType newType) {
  Attribute value = constant.getValueAttr();
  if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
    constant.setValueAttr(dense.reshape(newType));
    return success();
  }
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(value)) {
    constant.setValueAttr(
        DenseResourceElementsAttr::get(newType, resource.getRawHandle()));
    return success();
  }
  if (auto sparse = dyn_cast<SparseElementsAttr>(value)) {
    constant.setValueAttr(
        SparseElementsAttr::get(newType, sparse.getIndices(),
                                sparse.getValues()));
    return success();
  }
  return constant.emitError()
         << "cannot retag tensor constant with unsupported value attribute "
         << value;
}

bool canRetagTensorResult(
    Operation *producer, Attribute desiredEncoding, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues) {
  if (!producer)
    return false;
  if (isa<arith::ConstantOp>(producer) ||
      isa<ttg::LocalLoadOp, tt::MakeRangeOp>(producer))
    return true;
  if (auto dot = dyn_cast<tt::DotOp>(producer)) {
    auto *accumulator =
        solver.lookupState<layout::TensorLayoutLattice>(dot.getC());
    return accumulator && !accumulator->getValue().isUninitialized() &&
           !accumulator->getValue().isUnknown() &&
           accumulator->getValue().getLayoutEncoding() == desiredEncoding;
  }

  Attribute requiredSource =
      ::mlir::inferSrcEncoding(producer, desiredEncoding);
  if (!requiredSource &&
      !gd::hasSameTensorEncodingRelation(producer) &&
      !layout::isTransparentLayoutCarrierOp(producer))
    return false;
  if (!requiredSource)
    requiredSource = desiredEncoding;

  for (Value operand : producer->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type)
      continue;
    if (!isAutoTensor(operand)) {
      if (layout::unwrapNoVerifyEncoding(type.getEncoding()) !=
          requiredSource)
        return false;
      continue;
    }
    // SSA producers are visited before their users. A still-Auto operand
    // therefore means its producer/carrier could not absorb the requirement.
    // Retagging only this result would violate the op's declared layout
    // relation, so preserve the residual contract as a conversion boundary.
    return false;
  }
  return true;
}

bool materializeProducerInputBoundaries(
    Value value, Attribute desiredEncoding, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues) {
  Operation *producer = value.getDefiningOp();
  if (!producer ||
      isa<tt::DotOp, ttg::ConvertLayoutOp, gd::SetAutoLayoutOp,
          gd::RequireLayoutOp, gd::ReleaseLayoutOp>(producer) ||
      isa<RegionBranchOpInterface>(producer))
    return false;

  Attribute requiredSource =
      ::mlir::inferSrcEncoding(producer, desiredEncoding);
  if (!requiredSource &&
      (gd::hasSameTensorEncodingRelation(producer) ||
       gd::hasSameLoadStoreTensorEncodingRelation(producer)))
    requiredSource = desiredEncoding;
  if (!requiredSource)
    return false;

  // All tensor results of an exact producer relation must agree before any
  // operand edge is rewritten.
  for (Value result : producer->getResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type || result == value)
      continue;
    Attribute current =
        layout::unwrapNoVerifyEncoding(type.getEncoding());
    if (!isAutoEncoding(current)) {
      if (current != desiredEncoding)
        return false;
      continue;
    }
    if (blockedValues.contains(result))
      return false;
    auto *state =
        solver.lookupState<layout::TensorLayoutLattice>(result);
    if (!state || state->getValue().isUninitialized() ||
        state->getValue().isUnknown() ||
        state->getValue().getLayoutEncoding() != desiredEncoding)
      return false;
  }

  OpBuilder builder(producer);
  for (OpOperand &operand : producer->getOpOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!type)
      continue;
    Attribute current =
        layout::unwrapNoVerifyEncoding(type.getEncoding());
    if (current == requiredSource)
      continue;
    auto requiredType = cloneTensor(type, requiredSource);
    Value converted = builder.create<ttg::ConvertLayoutOp>(
        producer->getLoc(), requiredType, operand.get());
    operand.set(converted);
  }
  return true;
}

LogicalResult rewriteTensorValue(
    Value value, DataFlowSolver &solver,
    const llvm::DenseSet<Value> &blockedValues) {
  if (!isAutoTensor(value) || blockedValues.contains(value))
    return success();
  auto *lattice = solver.lookupState<layout::TensorLayoutLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized() ||
      lattice->getValue().isUnknown())
    return success();

  Attribute encoding = lattice->getValue().getLayoutEncoding();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    SmallVector<Value> predecessors;
    if (!layout::appendRegionCarrierPredecessors(value, predecessors))
      return success();
  } else if (!canRetagTensorResult(value.getDefiningOp(), encoding, solver,
                                   blockedValues) &&
             !materializeProducerInputBoundaries(value, encoding, solver,
                                                 blockedValues)) {
    return success();
  }

  auto newType =
      cloneTensor(cast<RankedTensorType>(value.getType()), encoding);
  if (newType == value.getType())
    return success();
  value.setType(newType);
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    return retagConstant(constant, newType);
  return success();
}

LogicalResult rewriteMemDescValue(Value value, DataFlowSolver &solver,
                                  Operation *diagnosticOp,
                                  const layout::GluonMemDescAliasAnalysis
                                      &aliases) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type)
    return success();
  auto *lattice =
      solver.lookupState<layout::LayoutEncodingLattice>(value);
  if (!lattice)
    return diagnosticOp->emitError()
           << "missing memdesc layout lattice for " << value;
  const layout::LayoutEncoding &state = lattice->getValue();
  if (state.isUninitialized() || state.isUnknown())
    return success();

  Attribute desired = state.getLayoutEncoding();
  Attribute current = layout::unwrapNoVerifyEncoding(type.getEncoding());
  if (current == desired || !isAutoMemDesc(value, aliases))
    return success();
  value.setType(cloneMemDesc(type, desired));
  return success();
}

void synchronizeFunctionInputs(tt::FuncOp func) {
  if (func.getBody().empty())
    return;
  SmallVector<Type> inputs(func.getBody().front().getArgumentTypes());
  func.setFunctionType(FunctionType::get(
      func.getContext(), inputs, func.getFunctionType().getResults()));
}

LogicalResult applyExplicitContracts(tt::FuncOp func) {
  if (!hasExplicitLayoutContract(func))
    return success();

  layout::GluonMemDescAliasAnalysis aliases(func);
  if (failed(aliases.initialize()))
    return failure();

  SymbolTableCollection symbols;
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<PathInsensitiveConstantPropagation>();
  solver.load<layout::LayoutBackwardPropagation>(symbols);
  solver.load<layout::LayoutForwardPropagation>();
  solver.load<layout::TensorBackwardPropagation>(symbols);
  if (failed(solver.initializeAndRun(func)))
    return failure();
  if (failed(layout::closeTensorLayoutRelations(func, solver)))
    return failure();

  llvm::DenseSet<Value> blocked =
      layout::computeBlockedTensorLayoutCarriers(func, solver);
  WalkResult rewritten = func.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) {
          if (isa<ttg::MemDescType>(argument.getType())) {
            if (failed(rewriteMemDescValue(argument, solver, op, aliases)))
              return WalkResult::interrupt();
          } else if (failed(rewriteTensorValue(argument, solver, blocked))) {
            return WalkResult::interrupt();
          }
        }

    for (Value result : op->getResults()) {
      if (isa<ttg::MemDescType>(result.getType())) {
        if (failed(rewriteMemDescValue(result, solver, op, aliases)))
          return WalkResult::interrupt();
      } else if (failed(rewriteTensorValue(result, solver, blocked))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (rewritten.wasInterrupted())
    return failure();

  updateTensorCarrierTypes(func, solver, blocked);
  synchronizeFunctionInputs(func);
  return success();
}

LogicalResult materializeResidualContract(Operation *op) {
  Value source = op->getOperand(0);
  Value result = op->getResult(0);
  bool sameConcreteType = source.getType() == result.getType();
  auto sourceTensor = dyn_cast<RankedTensorType>(source.getType());
  auto resultTensor = dyn_cast<RankedTensorType>(result.getType());
  if (sourceTensor && resultTensor &&
      sourceTensor.getShape() == resultTensor.getShape() &&
      sourceTensor.getElementType() == resultTensor.getElementType() &&
      layout::unwrapNoVerifyEncoding(sourceTensor.getEncoding()) ==
          layout::unwrapNoVerifyEncoding(resultTensor.getEncoding()))
    sameConcreteType = true;
  if (sameConcreteType) {
    result.setType(source.getType());
    result.replaceAllUsesWith(source);
    op->erase();
    return success();
  }

  if (isa<ttg::MemDescType>(result.getType()))
    return op->emitError()
           << "non-identity memdesc layout requirement remained after "
              "propagation; the selected candidate is incompatible with a "
              "fixed or opaque shared-memory view";

  OpBuilder builder(op);
  Value converted = builder.create<ttg::ConvertLayoutOp>(
      op->getLoc(), result.getType(), source);
  result.replaceAllUsesWith(converted);
  op->erase();
  return success();
}

LogicalResult cleanupContracts(tt::FuncOp func) {
  SmallVector<Operation *> contracts;
  func.walk([&](Operation *op) {
    if (isa<gd::SetAutoLayoutOp, gd::RequireLayoutOp,
            gd::ReleaseLayoutOp>(op))
      contracts.push_back(op);
  });
  for (Operation *op : contracts)
    if (failed(materializeResidualContract(op)))
      return failure();
  return success();
}

LogicalResult verifyNoContractOps(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (!isa<gd::SetAutoLayoutOp, gd::RequireLayoutOp,
             gd::ReleaseLayoutOp>(op))
      return WalkResult::advance();
    op->emitError() << "unresolved Gluon layout contract after propagation";
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

LogicalResult propagateLayouts(ModuleOp module) {
  if (failed(layout::verifyNoResidualCalls(
          module, "MetaX Gluon layout propagation")))
    return failure();
  for (tt::FuncOp func : module.getOps<tt::FuncOp>())
    if (failed(applyExplicitContracts(func)) ||
        failed(cleanupContracts(func)))
      return failure();
  return verifyNoContractOps(module);
}

class TritonMETAXGPUGluonPropagateLayoutPass
    : public impl::TritonMETAXGPUGluonPropagateLayoutBase<
          TritonMETAXGPUGluonPropagateLayoutPass> {
public:
  void runOnOperation() override {
    if (failed(propagateLayouts(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTritonMETAXGPUGluonPropagateLayoutPass() {
  return std::make_unique<TritonMETAXGPUGluonPropagateLayoutPass>();
}

} // namespace mlir

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
