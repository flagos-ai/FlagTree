#include "Gluon/Analysis/GluonDotAnalysis.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace {

bool isTensor(Value value) {
  return value && isa<RankedTensorType>(value.getType());
}

struct OperandPath {
  DotOperandPathFacts facts;
  SmallVector<Value, 2> terminals;
  SmallVector<Operation *, 1> permutations;
};

Value registerViewSource(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return {};
  return TypeSwitch<Operation *, Value>(def)
      .Case<tt::TransOp, tt::ExpandDimsOp, ttg::ConvertLayoutOp,
            gd::RequireLayoutOp, gd::ReleaseLayoutOp>(
          [](auto op) { return op.getSrc(); })
      .Case<ttg::ExtractTensorOp>(
          [](ttg::ExtractTensorOp op) { return op.getSource(); })
      .Default([](Operation *) { return Value{}; });
}

bool appendTensorLineagePredecessors(
    Operation *op, SmallVectorImpl<Value> &predecessors) {
  if (!op)
    return false;
  return TypeSwitch<Operation *, bool>(op)
      .Case<tt::TransOp, tt::ExpandDimsOp, ttg::ConvertLayoutOp,
            gd::RequireLayoutOp, gd::ReleaseLayoutOp>(
          [&](auto view) {
            predecessors.push_back(view.getSrc());
            return true;
          })
      .Case<tt::ReduceOp>([&](tt::ReduceOp reduce) {
        llvm::append_range(predecessors, reduce.getSrcs());
        return true;
      })
      .Case<ttg::ExtractTensorOp>([&](ttg::ExtractTensorOp extract) {
        predecessors.push_back(extract.getSource());
        return true;
      })
      .Default([&](Operation *operation) {
        if (!gd::hasSameTensorEncodingRelation(operation))
          return false;
        llvm::copy_if(operation->getOperands(),
                      std::back_inserter(predecessors), isTensor);
        return true;
      });
}

FailureOr<OperandPath>
analyzeOperandPath(Value root, tt::DotOp dot,
                   const GluonRegionBranchAnalysis &carriers) {
  struct Item {
    Value value;
    bool permuted;
  };
  SmallVector<Item, 8> worklist{{root, false}};
  DenseSet<Value> directVisited;
  DenseSet<Value> permutedVisited;
  llvm::SetVector<Value> terminals;
  llvm::SetVector<Operation *> permutations;
  bool sawDirect = false;
  bool sawPermuted = false;

  while (!worklist.empty()) {
    Item item = worklist.pop_back_val();
    DenseSet<Value> &visited =
        item.permuted ? permutedVisited : directVisited;
    if (!item.value || !visited.insert(item.value).second)
      continue;
    if (item.value.getDefiningOp<ttg::LocalLoadOp>()) {
      terminals.insert(item.value);
      (item.permuted ? sawPermuted : sawDirect) = true;
      continue;
    }
    if (auto permutation = item.value.getDefiningOp<ttg::BsmPermOp>()) {
      permutations.insert(permutation);
      worklist.push_back({permutation.getSrc1(), true});
      continue;
    }
    ArrayRef<Value> incoming = carriers.getPredecessors(item.value);
    if (!incoming.empty()) {
      for (Value value : incoming)
        worklist.push_back({value, item.permuted});
      continue;
    }
    if (Value source = registerViewSource(item.value)) {
      worklist.push_back({source, item.permuted});
      continue;
    }
    Operation *def = item.value.getDefiningOp();
    if (gd::hasSameTensorEncodingRelation(def)) {
      bool found = false;
      for (Value operand : def->getOperands())
        if (isTensor(operand)) {
          worklist.push_back({operand, item.permuted});
          found = true;
        }
      if (found)
        continue;
    }
    terminals.insert(item.value);
    (item.permuted ? sawPermuted : sawDirect) = true;
  }

  if (sawDirect && sawPermuted)
    return dot.emitError()
           << "dot operand merges direct and BSM-permuted paths";
  OperandPath result;
  result.facts.passesThroughBsmPermutation = sawPermuted;
  result.terminals.append(terminals.begin(), terminals.end());
  result.permutations.append(permutations.begin(), permutations.end());
  return result;
}

} // namespace

FailureOr<SmallVector<DotPipelineFacts, 4>>
collectDotPipelineFacts(ModuleOp module,
                        const GluonMemDescAliasAnalysis &aliases) {
  GluonRegionBranchAnalysis carriers(module);
  if (failed(carriers.initialize()))
    return failure();

  SmallVector<DotPipelineFacts, 4> pipelines;
  WalkResult walk = module.walk([&](tt::DotOp dot) {
    DotPipelineFacts pipeline;
    pipeline.dot = dot;
    for (unsigned operandIndex : {0u, 1u}) {
      DotPipelineOperandFacts &operandFacts =
          pipeline.operands[operandIndex];
      Value operand = operandIndex == 0 ? dot.getA() : dot.getB();
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || operandType.getRank() < 2) {
        dot.emitError()
            << "dot pipeline facts require rank-two-or-higher operands";
        return WalkResult::interrupt();
      }
      operandFacts.logicalKDimension = static_cast<unsigned>(
          operandType.getRank() - (operandIndex == 0 ? 1 : 2));

      FailureOr<OperandPath> path =
          analyzeOperandPath(operand, dot, carriers);
      if (failed(path))
        return WalkResult::interrupt();
      operandFacts.path = path->facts;
      operandFacts.terminals = std::move(path->terminals);
      operandFacts.bsmPermutations = std::move(path->permutations);

      for (Value terminal : operandFacts.terminals) {
        auto localLoad = terminal.getDefiningOp<ttg::LocalLoadOp>();
        if (!localLoad)
          continue;
        FailureOr<MemDescAliasPath> sharedView =
            aliases.getPath(localLoad.getSrc());
        if (failed(sharedView)) {
          localLoad.emitError()
              << "dot local_load has no proven logical layout-family path";
          return WalkResult::interrupt();
        }
        operandFacts.localLoads.push_back(
            {localLoad, std::move(*sharedView)});
      }

      SmallVector<Value, 8> lineage{operand};
      DenseSet<Value> visited;
      while (!lineage.empty()) {
        Value value = lineage.pop_back_val();
        if (!value || !visited.insert(value).second)
          continue;
        if (value.getDefiningOp<tt::DotOp>()) {
          operandFacts.hasDotProducer = true;
          break;
        }
        ArrayRef<Value> incoming = carriers.getPredecessors(value);
        if (!incoming.empty())
          llvm::append_range(lineage, incoming);
        else
          appendTensorLineagePredecessors(value.getDefiningOp(), lineage);
      }
    }
    pipelines.push_back(std::move(pipeline));
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  return pipelines;
}

} // namespace mlir::triton::gpu::metax::gluon
