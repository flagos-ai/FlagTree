#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "metax-gluon-region-branch-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu::metax::gluon {
namespace {

bool sameSuccessor(const RegionSuccessor &lhs, const RegionSuccessor &rhs) {
  return lhs.isParent() == rhs.isParent() &&
         lhs.getSuccessor() == rhs.getSuccessor() &&
         llvm::equal(lhs.getSuccessorInputs(), rhs.getSuccessorInputs());
}

} // namespace

static void collectRegionBranchSuccessors(
    RegionBranchOpInterface branch,
    SmallVectorImpl<RegionSuccessor> &result) {
  auto append = [&](ArrayRef<RegionSuccessor> candidates) {
    for (RegionSuccessor candidate : candidates)
      if (llvm::none_of(result, [&](const RegionSuccessor &current) {
            return sameSuccessor(current, candidate);
          }))
        result.push_back(candidate);
  };
  SmallVector<RegionSuccessor> candidates;
  branch.getSuccessorRegions(RegionBranchPoint::parent(), candidates);
  append(candidates);
  for (Region &region : branch->getRegions()) {
    candidates.clear();
    branch.getSuccessorRegions(region, candidates);
    append(candidates);
  }
}

LogicalResult GluonRegionBranchAnalysis::initialize() {
  edges.clear();
  predecessors.clear();
  successors.clear();

  root->walk([&](RegionBranchOpInterface branch) {
    SmallVector<RegionSuccessor> regionSuccessors;
    collectRegionBranchSuccessors(branch, regionSuccessors);
    auto reachesTarget = [&](RegionBranchPoint source,
                             const RegionSuccessor &target) {
      SmallVector<RegionSuccessor> candidates;
      branch.getSuccessorRegions(source, candidates);
      return llvm::any_of(candidates, [&](const RegionSuccessor &candidate) {
        return sameSuccessor(candidate, target);
      });
    };
    auto appendEdges = [&](Region *predecessorRegion,
                           const RegionSuccessor &successor,
                           ValueRange incoming) {
      Region *successorRegion =
          successor.isParent() ? nullptr : successor.getSuccessor();
      for (auto [inputIndex, successorInput] :
           llvm::enumerate(successor.getSuccessorInputs())) {
        if (inputIndex >= incoming.size())
          continue;
        Value predecessor = incoming[inputIndex];
        if (predecessor == successorInput)
          continue;

        auto &stored = predecessors[successorInput];
        if (!llvm::is_contained(stored, predecessor))
          stored.push_back(predecessor);
        auto &storedSuccessors = this->successors[predecessor];
        if (!llvm::is_contained(storedSuccessors, successorInput))
          storedSuccessors.push_back(successorInput);
        bool duplicate = llvm::any_of(edges, [&](const RegionCarrierEdge &edge) {
          return edge.owner == branch.getOperation() &&
                 edge.predecessorRegion == predecessorRegion &&
                 edge.successorRegion == successorRegion &&
                 edge.predecessor == predecessor &&
                 edge.successorInput == successorInput;
        });
        if (duplicate)
          continue;
        edges.push_back(RegionCarrierEdge{
            branch.getOperation(), predecessorRegion, successorRegion,
            predecessor, successorInput});
        LDBG("[region-carrier] owner="
             << branch->getName() << " source="
             << (predecessorRegion ? "region" : "parent") << " target="
             << (successorRegion ? "region" : "parent")
             << " input=" << inputIndex << " predecessor=" << predecessor
             << " successor=" << successorInput);
      }
    };

    for (const RegionSuccessor &successor : regionSuccessors) {
      RegionBranchPoint target(successor);
      if (reachesTarget(RegionBranchPoint::parent(), successor))
        appendEdges(/*predecessorRegion=*/nullptr, successor,
                    branch.getEntrySuccessorOperands(target));

      for (Region &region : branch->getRegions()) {
        if (!reachesTarget(RegionBranchPoint(region), successor))
          continue;
        for (Block &block : region) {
          auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
              block.getTerminator());
          if (!terminator)
            continue;
          appendEdges(&region, successor,
                      terminator.getSuccessorOperands(target));
        }
      }
    }
  });
  return success();
}

ArrayRef<Value>
GluonRegionBranchAnalysis::getPredecessors(Value value) const {
  auto it = predecessors.find(value);
  if (it == predecessors.end())
    return {};
  return it->second;
}

ArrayRef<Value>
GluonRegionBranchAnalysis::getSuccessors(Value value) const {
  auto it = successors.find(value);
  if (it == successors.end())
    return {};
  return it->second;
}

bool appendRegionCarrierPredecessors(
    Value value, SmallVectorImpl<Value> &predecessors) {
  RegionBranchOpInterface owner;
  if (auto argument = dyn_cast<BlockArgument>(value))
    owner = dyn_cast_or_null<RegionBranchOpInterface>(
        argument.getOwner()->getParentOp());
  else if (auto result = dyn_cast<OpResult>(value))
    owner = dyn_cast<RegionBranchOpInterface>(result.getOwner());
  if (!owner)
    return false;
  GluonRegionBranchAnalysis carriers(owner);
  if (failed(carriers.initialize()))
    return false;
  llvm::append_range(predecessors, carriers.getPredecessors(value));
  return true;
}

} // namespace mlir::triton::gpu::metax::gluon
