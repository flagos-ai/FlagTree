/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "TritonToGraph/GraphAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "graph-analysis"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// CFGTraverser Implementation
//===----------------------------------------------------------------------===//

void CFGTraverser::dfsForward(CFGTraversalBase &visitor) {
  DenseSet<BasicBlock *> visited;
  TraversalContext ctx;
  dfsForwardImpl(cfg.getEntryBlock(), visited, ctx, visitor);
}

void CFGTraverser::dfsForward(BasicBlock *start, CFGTraversalBase &visitor) {
  DenseSet<BasicBlock *> visited;
  TraversalContext ctx;
  dfsForwardImpl(start, visited, ctx, visitor);
}

void CFGTraverser::dfsForwardImpl(BasicBlock *block,
                                  DenseSet<BasicBlock *> &visited,
                                  TraversalContext &ctx,
                                  CFGTraversalBase &visitor) {
  if (!block || visited.contains(block))
    return;

  visited.insert(block);

  // pre-visit block
  visitor.preVisitBlock(block, ctx);

  // visit instructions in block
  for (auto &instPtr : block->getInstructions()) {
    Instruction *inst = instPtr.get();

    // check for structure entry
    if (inst->hasSubGraph()) {
      visitor.onEnterStructure(block, ctx);
      ctx.push(block);
    }

    visitor.VisitInstruction(inst, ctx);

    // check for structure exit (last instruction in structure block)
    if (inst->hasSubGraph()) {
      ctx.pop();
      visitor.onExitStructure(block, ctx);
    }
  }

  // visit successors
  for (BasicBlock *succ : block->getSuccessors()) {
    if (visited.contains(succ))
      continue;

    // check for back edge
    if (cfg.isBackEdge(block, succ)) {
      visitor.onBackEdge(block, succ, ctx);
    }

    dfsForwardImpl(succ, visited, ctx, visitor);
  }

  visitor.postVisitBlock(block, ctx);
}

void CFGTraverser::dfsBackward(BasicBlock *start, CFGTraversalBase &visitor) {
  DenseSet<BasicBlock *> visited;
  TraversalContext ctx;
  dfsBackwardImpl(start, visited, ctx, visitor);
}

void CFGTraverser::dfsBackwardImpl(BasicBlock *block,
                                   DenseSet<BasicBlock *> &visited,
                                   TraversalContext &ctx,
                                   CFGTraversalBase &visitor) {
  if (!block || visited.contains(block))
    return;

  visited.insert(block);

  visitor.preVisitBlock(block, ctx);

  // visit instructions in reverse order
  auto &insts = block->getInstructions();
  for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
    Instruction *inst = it->get();
    visitor.VisitInstruction(inst, ctx);
  }

  // visit predecessors
  for (BasicBlock *pred : block->getPredecessors()) {
    if (cfg.isBackEdge(pred, block)) {
      visitor.onBackEdge(pred, block, ctx);
    }
    dfsBackwardImpl(pred, visited, ctx, visitor);
  }

  visitor.postVisitBlock(block, ctx);
}

void CFGTraverser::bfsForward(CFGTraversalBase &visitor) {
  bfsForward(cfg.getEntryBlock(), visitor);
}

void CFGTraverser::bfsForward(BasicBlock *start, CFGTraversalBase &visitor) {
  DenseSet<BasicBlock *> visited;
  SmallVector<std::pair<BasicBlock *, TraversalContext>> worklist;

  worklist.push_back({start, TraversalContext()});
  visited.insert(start);

  while (!worklist.empty()) {
    auto [block, ctx] = worklist.pop_back_val();

    preVisitBlock(block, const_cast<TraversalContext &>(ctx));

    for (auto &instPtr : block->getInstructions()) {
      Instruction *inst = instPtr.get();
      visitor.VisitInstruction(inst, const_cast<TraversalContext &>(ctx));
    }

    visitor.postVisitBlock(block, const_cast<TraversalContext &>(ctx));

    for (BasicBlock *succ : block->getSuccessors()) {
      if (!visited.contains(succ)) {
        visited.insert(succ);
        worklist.push_back({succ, ctx});
      }
    }
  }
}

void CFGTraverser::bfsBackward(BasicBlock *start, CFGTraversalBase &visitor) {
  DenseSet<BasicBlock *> visited;
  SmallVector<std::pair<BasicBlock *, TraversalContext>> worklist;

  worklist.push_back({start, TraversalContext()});
  visited.insert(start);

  while (!worklist.empty()) {
    auto [block, ctx] = worklist.pop_back_val();

    visitor.preVisitBlock(block, const_cast<TraversalContext &>(ctx))

        for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
      Instruction *inst = it->get();
      visitor.VisitInstruction(inst, ctx);
    }

    visitor.postVisitBlock(block, const_cast<TraversalContext &>(ctx));

    for (BasicBlock *pred : block->getPredecessors()) {
      if (!visited.contains(pred)) {
        visited.insert(pred);
        worklist.push_back({pred, ctx});
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// DFGTraverser Implementation
//===----------------------------------------------------------------------===//

void DFGTraverser::dfsBackward(Value seed, DFGTraversalBase &visitor,
                               const Options &opts) {
  DenseSet<Operation *> visited;
  dfsBackwardImpl(seed, visitor, visited, pts, 0);
}

void DFGTraverser::dfsBackward(ArrayRef<Value> seeds, DFGTraversalBase &visitor,
                               const Options &opts) {
  DenseSet<Operation *> visited;
  for (Value seed : seeds) {
    dfsBackwardImpl(seed, visitor, visited, opts, 0);
  }
}

void DFGTraverser::dfsBackwardImpl(Value value, DFGTraversalBase &visitor,
                                   DenseSet<Operation *> &visited,
                                   const Options &opts, int depth) {
  if (opts.maxDepth >= 0 && depth > opts.maxDepth)
    return;

  // get definition
  Operation *defOp = nullptr;
  if (opts.useMemorySSA) {
    auto result = dfg.queryDataFlow(value);
    if (auto *memResult = dyn_cast<MemorySSAResult>(result.get())) {
      defOp = memResult->getDefinition()->getDefOp();
    }
  } else {
    defOp = value.getDefiningOp();
  }

  if (!defOp)
    return;

  if (visited.contains(defOp))
    return;

  if (opts.stopOps.contains(defOp))
    return;

  visited.insert(defOp);

  visitor.VisitDef(value, defOp, depth);

  // recursively visit operands
  for (Value operand : defOp->getOperands()) {
    dfsBackwardImpl(operand, visitor, visited, opts, depth + 1);
  }

  // handle phi/iter_arg
  if (opts.followPhi) {
    auto &dataFlowInfo = dfg.getDataFlowInfo();
    if (dataFlowInfo.hasPhi(value)) {
      auto &phiInfo = dataFlowInfo.getPhi(value);
      visitor.onPhi(value, phiInfo, depth);
    }
  }
}

void DFGTraverser::dfsForward(Value seed, DFGTraversalBase &visitor,
                              const Options &opts) {
  DenseSet<Operation *> visited;
  dfsForwardImpl(seed, visitor, visited, ctx, opts, 0);
}

void DFGTraverser::dfsForwardImpl(Value value, DFGTraversalBase &visitor,
                                  DenseSet<Operation *> &visited,
                                  const Options &opts, int depth) {
  if (opts.maxDepth >= 0 && depth > opts.maxDepth)
    return;

  SmallVector<OpOperand *> uses;
  if (opts.useMemorySSA) {
    uses = dfg.getDataFlowInfo().getSSAUses(value);
  } else {
    for (OpOperand &use : value.getUses()) {
      uses.push_back(&use);
    }
  }

  for (OpOperand *use : uses) {
    Operation *userOp = use->getOwner();

    if (visited.contains(userOp))
      continue;

    if (opts.stopOps.contains(userOp))
      continue;

    visited.insert(userOp);

    visitor.VisitUse(value, use, depth);

    // visit results of this operation
    for (Value result : userOp->getResults()) {
      dfsForwardImpl(result, visitor, visited, opts, depth + 1);
    }
  }
}

//===----------------------------------------------------------------------===//
// Region Implementation
//===----------------------------------------------------------------------===//

void Region::add(Instruction *inst) { instSet_.insert(inst); }

void Region::add(Operation *op, ControlFlowGraph &cfg) {
  if (Instruction *inst = cfg.getInstruction(op)) {
    add(inst);
  }
}

void Region::addAll(ArrayRef<Instruction *> insts) {
  for (Instruction *inst : insts) {
    add(inst);
  }
}

bool Region::contains(Instruction *inst) const {
  return instSet_.contains(inst);
}

bool Region::contains(Operation *op) const {
  // note: requires cfg to be available - caller must ensure this
  // use the version that takes cfg as parameter for proper lookup
  return false;
}

void Region::remove(Instruction *inst) { instSet_.erase(inst); }

void Region::clear() { instSet_.clear(); }

SmallVector<Instruction *> Region::orderedInstructions() const {
  SmallVector<Instruction *> result(instSet_.begin(), instSet_.end());
  // sort by block id then instruction index
  llvm::sort(result, [](Instruction *a, Instruction *b) {
    auto *bbA = a->getParentBlock();
    auto *bbB = b->getParentBlock();
    if (bbA != bbB)
      return bbA->getId() < bbB->getId();
    // within same block, find position
    // this is approximate - exact order requires full block scan
    return a->getId() < b->getId();
  });
  return result;
}

SmallVector<Operation *> Region::operations() const {
  SmallVector<Operation *> result;
  for (Instruction *inst : instSet_) {
    if (Operation *op = inst->getOperation()) {
      result.push_back(op);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// RegionAnalyzer Implementation
//===----------------------------------------------------------------------===//

bool RegionAnalyzer::hasDependency(const Region &from, const Region &to) const {
  auto deps = getDependencies(from, to);
  return !deps.empty();
}

SmallVector<RegionAnalyzer::Dependency>
RegionAnalyzer::getDependencies(const Region &from, const Region &to) const {
  SmallVector<Dependency> deps;

  // check for data dependencies: from defines, to uses
  for (Instruction *fromInst : from) {
    for (Value result : fromInst->getOperation()->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *userOp = use.getOwner();
        if (Instruction *toInst = cfg.getInstruction(userOp)) {
          if (to.contains(toInst)) {
            deps.push_back({Dependency::DATA, result, fromInst, toInst});
          }
        }
      }
    }
  }

  return deps;
}

RegionAnalyzer::ExternalDeps
RegionAnalyzer::analyzeExternalDeps(const Region &region) const {
  ExternalDeps result;

  // find inputs: external definitions used inside region
  for (Instruction *inst : region) {
    for (Value operand : inst->getOperation()->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        // block argument - treat as external input
        bool alreadyTracked = false;
        for (auto &input : result.inputs) {
          if (input.value == operand) {
            input.internalUses.push_back(inst);
            alreadyTracked = true;
            break;
          }
        }
        if (!alreadyTracked) {
          result.inputs.push_back({operand, nullptr, {inst}});
        }
      } else if (Instruction *defInst = cfg.getInstruction(defOp)) {
        if (!region.contains(defInst)) {
          // external definition
          bool alreadyTracked = false;
          for (auto &input : result.inputs) {
            if (input.value == operand) {
              input.internalUses.push_back(inst);
              alreadyTracked = true;
              break;
            }
          }
          if (!alreadyTracked) {
            result.inputs.push_back({operand, defInst, {inst}});
          }
        }
      }
    }
  }

  // find outputs: internal definitions used outside region
  for (Instruction *inst : region) {
    for (Value result : inst->getOperation()->getResults()) {
      SmallVector<Instruction *> externalUses;
      for (OpOperand &use : result.getUses()) {
        Operation *userOp = use.getOwner();
        if (Instruction *userInst = cfg.getInstruction(userOp)) {
          if (!region.contains(userInst)) {
            externalUses.push_back(userInst);
          }
        }
      }
      if (!externalUses.empty()) {
        result.outputs.push_back({result, inst, externalUses});
      }
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// ProgramSlicer Implementation
//===----------------------------------------------------------------------===//

ProgramSlice ProgramSlicer::compute(const SliceCriterion &criterion) {
  ProgramSlice slice;

  DFGTraverser dfgTraverser(dfg);

  for (Value seed : criterion.seeds) {
    class SliceBuilder : public DFGTraversalBase {
    public:
      SliceBuilder(ProgramSlice &slice, ControlFlowGraph &cfg)
          : slice(slice), cfg(cfg) {}

      bool VisitDef(Value value, Operation *defOp, int depth) override {
        if (Instruction *inst = cfg.getInstruction(defOp)) {
          slice.add(inst);
        }
        return true;
      }

      bool VisitUse(Value value, OpOperand *use, int depth) override {
        Operation *userOp = use->getOwner();
        if (Instruction *inst = cfg.getInstruction(userOp)) {
          slice.add(inst);
        }
        return true;
      }

    private:
      ProgramSlice &slice;
      ControlFlowGraph &cfg;
    };

    SliceBuilder builder(slice, cfg);

    switch (criterion.dir) {
    case SliceCriterion::BACKWARD:
      dfgTraverser.dfsBackward(seed, builder, criterion.dfgOpts);
      break;
    case SliceCriterion::FORWARD:
      dfgTraverser.dfsForward(seed, builder, criterion.dfgOpts);
      break;
    case SliceCriterion::BIDIRECTIONAL:
      dfgTraverser.dfsBackward(seed, builder, criterion.dfgOpts);
      dfgTraverser.dfsForward(seed, builder, criterion.dfgOpts);
      break;
    }
  }

  return slice;
}

ProgramSlice ProgramSlicer::sliceFromYields(ArrayRef<Value> yields,
                                            SliceCriterion::Direction dir) {
  SliceCriterion criterion;
  criterion.seeds = SmallVector<Value>(yields);
  criterion.dir = dir;
  return compute(criterion);
}

void ProgramSlice::merge(const ProgramSlice &other) {
  for (Instruction *inst : other) {
    instructions_.insert(inst);
  }
}

void ProgramSlice::intersect(const ProgramSlice &other) {
  DenseSet<Instruction *> toRemove;
  for (Instruction *inst : instructions_) {
    if (!other.contains(inst)) {
      toRemove.insert(inst);
    }
  }
  for (Instruction *inst : toRemove) {
    instructions_.erase(inst);
  }
}

void ProgramSlice::subtract(const ProgramSlice &other) {
  for (Instruction *inst : other) {
    instructions_.erase(inst);
  }
}

Region ProgramSlice::toRegion(StringRef name) const {
  Region region(name);
  for (Instruction *inst : instructions_) {
    region.add(inst);
  }
  return region;
}

//===----------------------------------------------------------------------===//
// RegionAbsorber Implementation
//===----------------------------------------------------------------------===//

void RegionAbsorber::absorb(Region &region, ArrayRef<Instruction *> seeds,
                            const AbsorptionPolicy &policy) {
  DenseSet<Instruction *> visited;

  for (Instruction *seed : seeds) {
    region.add(seed);

    if (policy.dir == AbsorptionPolicy::UPSTREAM ||
        policy.dir == AbsorptionPolicy::BOTH) {
      absorbUpstream(region, seed, policy, visited, 0);
    }

    if (policy.dir == AbsorptionPolicy::DOWNSTREAM ||
        policy.dir == AbsorptionPolicy::BOTH) {
      absorbDownstream(region, seed, policy, visited, 0);
    }
  }
}

void RegionAbsorber::absorbUpstream(Region &region, Instruction *inst,
                                    const AbsorptionPolicy &policy,
                                    DenseSet<Instruction *> &visited,
                                    int depth) {
  if (policy.maxDepth >= 0 && depth > policy.maxDepth)
    return;

  if (visited.contains(inst))
    return;
  visited.insert(inst);

  if (policy.shouldStop && policy.shouldStop(inst))
    return;

  if (policy.stopOps.contains(inst->getOperation()))
    return;

  // add to region
  region.add(inst);

  // visit operands
  for (Value operand : inst->getOperation()->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      if (Instruction *defInst = cfg.getInstruction(defOp)) {
        if (!policy.crossRegionBoundary && region.contains(defInst))
          continue;

        absorbUpstream(region, defInst, policy, visited, depth + 1);
      }
    }
  }
}

void RegionAbsorber::absorbDownstream(Region &region, Instruction *inst,
                                      const AbsorptionPolicy &policy,
                                      DenseSet<Instruction *> &visited,
                                      int depth) {
  if (policy.maxDepth >= 0 && depth > policy.maxDepth)
    return;

  if (visited.contains(inst))
    return;
  visited.insert(inst);

  if (policy.shouldStop && policy.shouldStop(inst))
    return;

  if (policy.stopOps.contains(inst->getOperation()))
    return;

  // add to region
  region.add(inst);

  // visit uses
  for (Value result : inst->getOperation()->getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *userOp = use.getOwner();
      if (Instruction *userInst = cfg.getInstruction(userOp)) {
        if (!policy.crossRegionBoundary && region.contains(userInst))
          continue;

        absorbDownstream(region, userInst, policy, visited, depth + 1);
      }
    }
  }
}

void RegionAbsorber::absorbFromValue(Region &region, Value value,
                                     const AbsorptionPolicy &policy) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return;

  Instruction *inst = cfg.getInstruction(defOp);
  if (!inst)
    return;

  absorb(region, {inst}, policy);
}

void RegionAbsorber::absorbUntilBoundary(
    Region &region, ArrayRef<Instruction *> seeds,
    std::function<bool(Instruction *)> isBoundary) {
  AbsorptionPolicy policy;
  policy.shouldStop = isBoundary;
  absorb(region, seeds, policy);
}
