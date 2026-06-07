#include "TritonAffinityOpt/DAG.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace mlir {
namespace AffinityDAG {

const auto printFlags =
    OpPrintingFlags().enableDebugInfo(true, true).skipRegions();

const char *literalCoreType(CoreType ct) {
  switch (ct) {
  case VECTOR_ONLY:
    return "VECTOR_ONLY";
  case CUBE_ONLY:
    return "CUBE_ONLY";
  case CUBE_AND_VECTOR:
    return "CUBE_AND_VECTOR";
  case UNDETERMINED:
    return "UNDETERMINED";
  }
  return "Unknown";
}

bool opIsScf(Operation *op) {
  if (!llvm::isa<scf::SCFDialect>(op->getDialect()))
    return false;
  return true;
}

Graph::Graph(Block *block, Graph *parent, OpMap opMap, ValueMap valueMap,
             bool inheritParent)
    : block(block), parent(parent), opMap(opMap), valueMap(valueMap) {

  if (parent && inheritParent) {
    if (!this->opMap) {
      this->opMap = parent->opMap;
    }

    if (!this->valueMap) {
      this->valueMap = parent->valueMap;
    }
  }

  if (!this->opMap) {
    this->opMap = std::make_shared<OpMapRaw>();
  }

  if (!this->valueMap) {
    this->valueMap = std::make_shared<ValueMapRaw>();
  }

  for (auto blockArg : block->getArguments()) {
    (*this->valueMap)[blockArg] = std::make_unique<ValueNode>(blockArg);
    blockArgs.push_back((*this->valueMap)[blockArg].get());
  }

  for (auto &opRef : block->getOperations()) {
    opCount += 1;
    auto op = &opRef;
    auto opNodeUnique = std::make_unique<OpNode>(op, this);
    auto opNode = opNodeUnique.get();
    (*this->opMap)[op] = std::move(opNodeUnique);

    if (block->mightHaveTerminator() && op == block->getTerminator()) {
      terminator = opNode;
    }

    for (auto &subgraph : opNode->subgraphs) {
      opCount += subgraph.opCount;
    }
  }
};

bool valueIsScalar(Value value) {
  auto type = value.getType();

  if (type.isIntOrIndexOrFloat()) {
    return true;
  }

  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() == 0;
  }

  if (auto _ = llvm::dyn_cast<triton::PointerType>(type)) {
    return true;
  }

  return false;
}

bool valueIsTensorOfPtr(Value value) {
  auto type = value.getType();
  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    auto elementType = tensorType.getElementType();
    if (llvm::isa<triton::PointerType>(elementType)) {
      return true;
    }
  }

  return false;
}

OpAbility OpNode::canRunOn() const {
  if (opIsScf(op)) {
    return OpAbility::CUBE_AND_VECTOR;
  }
  return llvm::TypeSwitch<Operation *, OpAbility>(op)
      .Case<triton::DotOp>([](auto) { return OpAbility::CUBE_ONLY; })
      .Case<arith::ConstantOp, triton::AdvanceOp, triton::TransOp,
            annotation::MarkOp>([](auto) { return OpAbility::CUBE_AND_VECTOR; })
      .Case<arith::SelectOp>([](arith::SelectOp op) {
        // when cond is vector, selectOp should be vector, otherwise scalar
        return (valueIsScalar(op.getCondition()) ? OpAbility::CUBE_AND_VECTOR
                                                 : OpAbility::PREFER_VECTOR);
      })
      .Default([](Operation *op) {
        auto isVector = false;
        for (auto operand : op->getOperands()) {
          if (!valueIsScalar(operand)) {
            // if (valueIsTensorOfPtr(operand)) {
            //   return SCALAR;
            // }
            isVector = true;
          }
        }

        for (auto result : op->getResults()) {
          if (!valueIsScalar(result)) {
            // if (valueIsTensorOfPtr(result)) {
            //   return SCALAR;
            // }
            isVector = true;
          }
        }

        if (isVector) {
          return OpAbility::PREFER_VECTOR;
        }

        return OpAbility::CUBE_AND_VECTOR;
      });
}

OpNode::OpNode(Operation *op, Graph *graph) : Node(Node::NK_Op), op(op) {
  if (op == nullptr) {
    return;
  }

  llvm::outs() << op << "\n";

  auto &valueMap = *graph->valueMap.get();
  auto &opMap = *graph->opMap.get();
  for (const auto operand : op->getOperands()) {
    auto valueNode = valueMap.at(operand).get();
    valueNode->outputs.push_back(this);
    inputs.push_back(valueNode);
  }

  for (const auto &result : op->getResults()) {
    auto valueNodeUnique = std::make_unique<ValueNode>(result);
    auto valueNode = valueNodeUnique.get();
    valueMap[result] = std::move(valueNodeUnique);
    valueNode->source = this;
    outputs.push_back(valueNode);
  }

  // if (!op->hasTrait<OpTrait::SingleBlock>()) {
  //   llvm::dbgs() << "Not building subgraph because op is not SingleBlock: "
  //   << op << '\n'; return;
  // }

  if (auto branchOp = llvm::dyn_cast<RegionBranchOpInterface>(op)) {

    OpNode *terminator = nullptr;
    llvm::SmallVector<std::pair<Region &, Graph &>, 2> validRegions;

    for (auto &region : branchOp->getRegions()) {
      if (region.getBlocks().empty())
        continue;
      subgraphs.emplace_back(&region.getBlocks().front(), graph);
      validRegions.emplace_back(region, subgraphs.back());
    }

    for (auto [region, subgraph] : validRegions) {
      SmallVector<RegionSuccessor, 2> succRegions;

      branchOp.getSuccessorRegions(region, succRegions);
      if (auto currTerminator = dyn_cast<RegionBranchTerminatorOpInterface>(
              subgraph.terminator->op)) {
        for (auto &succ : succRegions) {
          auto forwardedVal = currTerminator.getSuccessorOperands(succ);
          if (succ.isParent()) {
            // Step1: first yield to parent -> results: double direction
            if (!terminator && subgraph.terminator) {
              terminator = subgraph.terminator;
              for (auto [forwardedVal, resultNode] :
                   llvm::zip_equal(forwardedVal, outputs)) {
                auto resultValueNode = llvm::dyn_cast<ValueNode>(resultNode);
                assert(resultValueNode &&
                       "Output of a OpNode should be ValueNode!");
                auto forwardedNode = valueMap[forwardedVal].get();
                resultValueNode->source = forwardedNode;
                forwardedNode->outputs.push_back(resultNode);
              }
            }

          } else {
            // Step2: Region terminator -> Succ Operands
            auto succRegion = succ.getSuccessor();

            for (auto [operand, succInput] :
                 llvm::zip_equal(forwardedVal, succ.getSuccessorInputs())) {
              auto forwardedNode = valueMap[operand].get();
              auto succNode = valueMap[succInput].get();
              forwardedNode->outputs.push_back(succNode);
              succNode->source = forwardedNode;
            }
          }
        }
      }
    }

    if (auto loopOp = llvm::dyn_cast<LoopLikeOpInterface>(op)) {
      // Step3: inits->iter_args (single directional) (should be handled in step
      // 2: ) last terminator -> iter_args (bidirectional)
      for (auto [init, iterArgVal] :
           llvm::zip_equal(loopOp.getInits(), loopOp.getRegionIterArgs())) {
        auto &initNode = valueMap[init];
        auto &iterArgNode = valueMap[iterArgVal];
        initNode->outputs.push_back(iterArgNode.get());
      }
      // for(auto [init, iterArgVal, yieldNode] :
      // llvm::zip_equal(loopOp.getInits(), loopOp.getRegionIterArgs(),
      // terminator->outputs)) {
      //   auto& initNode = valueMap[init];
      //   auto& iterArgNode = valueMap[iterArgVal];
      //   initNode->outputs.push_back(iterArgNode.get());
      //   yieldNode->outputs.push_back(iterArgNode.get());
      //   iterArgNode->source = yieldNode;
      // }
    }
  }
}

// llvm::SmallVector<ValueNode*, 4> getWriteOperandPriority(OpNode* op) {

//   llvm::SmallVector<ValueNode*, 4> result(op->getInputs());

//   auto getPriority = [](ValueNode* node) {
//     auto typ = getElementTypeOrSelf(node->value);
//     if (typ.isInteger(1)) {
//       return 2;
//     }
//     if (llvm::isa<triton::PointerType>(typ)) {
//       return 1;
//     }
//     return 0;
//   };

//   std::stable_sort(result.begin(), result.end(), [&](ValueNode* a, ValueNode*
//   b) {
//     return getPriority(a) < getPriority(b);
//   });

//   return result;
// }

ValueNode *getWriteDataSource(OpNode *op) {
  auto inputRange = op->getInputs();
  for (auto node : inputRange.drop_front()) {
    auto typ = getElementTypeOrSelf(node->value);
    if (!typ.isInteger(1)) {
      return node;
    }
  };

  return nullptr;
}

enum class MemPolicy { NONE, READ, WRITE };

CoreType Node::absorbCommon() {

  auto sourceNode = getSourceOpNode();
  auto op = sourceNode ? sourceNode->op : nullptr;

  if (!sourceNode || !op) {
    CoreType newCoreType = isOnPrivate;
    for (auto output : outputs) {
      newCoreType = newCoreType | output->isOn();
      isUpstreamOfCubeMem = isUpstreamOfCubeMem || output->isUpstreamOfCubeMem;
    }
    return newCoreType;
  }

  CoreType newCoreType = sourceNode->isOn();

  OpAbility ability = sourceNode->canRunOn();

  if (ability == OpAbility::CUBE_ONLY) {
    return CUBE_ONLY;
  }

  auto memIface = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  auto memPolicy = MemPolicy::NONE;

  if (memIface) {
    // Possible improvements: Determine the policy to use based on shapes,
    // inputs and outputs, etc
    if (memIface.hasEffect<MemoryEffects::Write>()) {
      memPolicy = MemPolicy::WRITE;
    } else if (memIface.hasEffect<MemoryEffects::Read>()) {
      memPolicy = MemPolicy::READ;
    }
  }

  if (memPolicy == MemPolicy::WRITE) {
    if (auto data = getWriteDataSource(sourceNode)) {
      auto currCt = data->isOn();
      if (exactlyOneType(currCt)) {
        if (currCt == CUBE_ONLY) {
          isUpstreamOfCubeMem = true;
        }
        return currCt;
      }
    }

    // data is not cube_only
    return VECTOR_ONLY;
  }

  for (auto output : outputs) {
    switch (output->isOn()) {
    case CUBE_AND_VECTOR:
      newCoreType = newCoreType | VECTOR_ONLY;
      // not breaking the switch because we need to handle cube
    case CUBE_ONLY:
      if (ability != OpAbility::PREFER_VECTOR || output->isUpstreamOfCubeMem ||
          memPolicy == MemPolicy::READ) {
        isUpstreamOfCubeMem =
            (isUpstreamOfCubeMem || output->isUpstreamOfCubeMem ||
             memPolicy == MemPolicy::READ);
        newCoreType = newCoreType | CUBE_ONLY;
      }
      break;
    case VECTOR_ONLY:
      newCoreType = newCoreType | VECTOR_ONLY;
    default: // UNDETERMINED, skip
      break;
    };
  }

  return newCoreType;
}

CoreType OpNode::absorbImpl() {
  if (opIsScf(op)) {
    return CUBE_AND_VECTOR;
  }

  auto newCoreType = absorbCommon();

  // if (canRunOn() == OpAbility::CUBE_AND_VECTOR) {
  //   for (auto input : inputs) {
  //     newCoreType = newCoreType | input->isOn();
  //   }
  // }

  return newCoreType;
}

CoreType ValueNode::absorbImpl() { return absorbCommon(); }

std::unique_ptr<Graph> Graph::fromMultiBlockFunc(triton::FuncOp funcOp) {

  auto dummyBlock = new Block();
  auto dummyGraph = std::make_unique<Graph>(dummyBlock);
  auto dummyNode = std::make_unique<OpNode>(nullptr, dummyGraph.get());
  size_t opCount = 0;

  for (auto &block : funcOp.getBody()) {
    auto &subgraph =
        dummyNode->subgraphs.emplace_back(&block, dummyGraph.get());
    opCount += subgraph.opCount;
  }

  auto &opMap = *dummyGraph->opMap.get();
  auto &valueMap = *dummyGraph->valueMap.get();

  llvm::SmallVector<Node *, 0> nodes;
  nodes.reserve(opMap.size() + valueMap.size());

  for (auto &[_, node] : opMap) {
    if (node.get())
      nodes.push_back(node.get());
  }

  for (auto &[_, node] : valueMap) {
    if (node.get())
      nodes.push_back(node.get());
  }

  auto diffuse = [&]() {
    // Not sure if determinism is required
    llvm::SmallSetVector<Node *, 16> worklist(nodes.begin(), nodes.end());

    size_t threshold = worklist.size() * 5;

    for (size_t i = 0; i < threshold; i++) {
      if (worklist.empty()) {
        break;
      }

      auto node = worklist.pop_back_val();

      if (node->absorb()) {
        auto affected = node->getAffected();
        worklist.insert(affected.begin(), affected.end());
      }
    }
  };

  diffuse();

  for (auto node : nodes) {
    if (node->isOn() == UNDETERMINED) {
      node->isOnPrivate = VECTOR_ONLY;
    }
  }

  diffuse();

  OpPrintingFlags flags;
  flags.skipRegions();

  for (auto [idx, node] : llvm::enumerate(nodes)) {
    llvm::TypeSwitch<Node *>(node)
        .Case<OpNode>([&, idx = idx](OpNode *node) {
          if (node->op) {
            llvm::dbgs() << llvm::formatv(
                "\n\n====== OpNode on: {1} @ {0} ======\n", node->op,
                literalCoreType(node->isOn()));
            node->op->print(llvm::dbgs(), flags);
            llvm::dbgs() << "\nAbility: "
                         << literalCoreType(toCoreType(node->canRunOn()));
            llvm::dbgs() << llvm::formatv("\n====== {0} ======\n", node->op);
          }
        })
        .Case<ValueNode>([&, idx = idx](ValueNode *node) {
          if (node->value) {
            llvm::dbgs() << llvm::formatv(
                "\n\n====== ValueNode on {1} @ {0} ======\n", node->value,
                literalCoreType(node->isOn()));
            node->value.print(llvm::dbgs(), flags);
            llvm::dbgs() << llvm::formatv("\n====== {0} ======\n", node->value);
          }
        });
    // if (auto opNode = llvm::dyn_cast<OpNode>(node)) {
    //   if (auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(opNode->op)) {
    //     llvm::dbgs() << "\n==== ForOp ====\n";
    //     llvm::dbgs() << forOp << "\n";
    //     llvm::dbgs() << "\n---- IterArgs ----\n";
    //     for(auto iterArg : forOp.getRegionIterArgs()) {
    //       auto& valueNode = valueMap[iterArg];
    //       llvm::dbgs() << llvm::formatv(
    //         "{0}: {1} upstream: {2} definingOp: {3} \n",
    //         iterArg.getArgNumber(),
    //         literalCoreType(valueNode->isOn()),
    //         literalCoreType(valueNode->source->isOn()),
    //         valueNode->getSourceOp()->op
    //       );
    //     }
    //     llvm::dbgs() << "\n---- Results ----\n";
    //     for(auto result : forOp.getResults()) {
    //       llvm::dbgs() << result.getResultNumber() << ' ' <<
    //       literalCoreType(valueMap[result]->isOn()) << '\n';
    //     }
    //   }
    // }
  }

  return dummyGraph;
};

} // namespace AffinityDAG
} // namespace mlir
