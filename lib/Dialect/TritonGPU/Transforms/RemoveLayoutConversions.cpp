#if __has_include("flagtree_spec.h")
#include "flagtree_spec.h"
#endif

#ifndef FLAGTREE_SPEC_Dialect_TritonGPU_Transforms_RemoveLayoutConversion

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#ifdef __FLAGTREE_RLC_ENHANCE__
#include "triton/Analysis/AxisInfo.h"
#endif // __FLAGTREE_RLC_ENHANCE__
#include "triton/Analysis/Utility.h"
#ifdef __TLE__
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif // __TLE__
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#ifdef __TLE__
#include "tle/dialect/include/Transforms/EncodingRematerialization.h"
#endif // __TLE__
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>
#ifdef __FLAGTREE_RLC_ENHANCE__
#include <optional>
#endif // __FLAGTREE_RLC_ENHANCE__

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUREMOVELAYOUTCONVERSIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-remove-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#ifdef __FLAGTREE_RLC_ENHANCE__
// Per-phase switches, AND-ed with the runtime master switch (rlcEnhance).
// Backward-propagation and small-component solving depend on cost-based
// resolution (disabling it forces both off); store-layout remat is independent.
static constexpr bool kEnableCostBasedResolution = true;
static constexpr bool kEnableBackwardPropagation = true;
static constexpr bool kEnableSmallComponentSolving = true;
static constexpr bool kEnableStoreLayoutRematerialization = true;
#endif // __FLAGTREE_RLC_ENHANCE__

namespace {

#ifdef __TLE__
static bool touchesTleRemotePointerPath(Value value, DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  StringRef opName = def->getName().getStringRef();
  if (opName == "tle.remote_pointers")
    return true;
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    unsigned idx = result.getResultNumber();
    return touchesTleRemotePointerPath(ifOp.thenYield().getOperand(idx),
                                       visited) ||
           touchesTleRemotePointerPath(ifOp.elseYield().getOperand(idx),
                                       visited);
  }
  for (Value operand : def->getOperands()) {
    if (touchesTleRemotePointerPath(operand, visited))
      return true;
  }
  return false;
}

#ifdef __FLAGTREE_RLC_ENHANCE__
// True if `value` sits on a TLE cluster remote-address chain, where cross-CTA
// addresses are built as tle.local_pointers -> tle.remote_pointers. The
// pointer/index tensors on that chain carry distributed-addressing semantics
// the cost model cannot see, so retagging them corrupts the remote access.
// Checks both the producer chain reaching a tle.remote_pointers and a use
// feeding one (a plain tle.local_pointers with no remote consumer is fine).
static bool valueOnTleRemotePointerPath(Value value) {
  DenseSet<Value> visited;
  if (touchesTleRemotePointerPath(value, visited))
    return true;
  if (Operation *def = value.getDefiningOp())
    if (def->getName().getStringRef() == "tle.remote_pointers")
      return true;
  for (Operation *user : value.getUsers()) {
    StringRef name = user->getName().getStringRef();
    if (name == "tle.remote_pointers")
      return true;
    if (name == "tle.local_pointers")
      for (Value result : user->getResults())
        for (Operation *ptrUser : result.getUsers())
          if (ptrUser->getName().getStringRef() == "tle.remote_pointers")
            return true;
  }
  return false;
}
#endif // __FLAGTREE_RLC_ENHANCE__
#endif // __TLE__

#ifdef __FLAGTREE_RLC_ENHANCE__
static Attribute getTensorEncoding(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType ? tensorType.getEncoding() : Attribute();
}

static bool allResultsHaveNoUses(Operation *op) {
  return llvm::all_of(op->getResults(),
                      [](Value result) { return result.use_empty(); });
}
#endif // __FLAGTREE_RLC_ENHANCE__

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// The current algorithm works by analyzing the IR and doing a one-shot rewrite
// based on the analysis. The algorithm is as follows.
//
// 1. Find all the anchor ops. These are ops that have a layout we want to
//    preserve.
//
// 2. For each anchor, propagate its layout to all its descendants.
//    An op can have multiple ancestors that are anchors, so at this stage an op
//    may have multiple layouts associated with it.
//
// 3. Resolve conflicts by deciding which of the multiple layouts the op should
//    keep, inserting convert-layout ops to resolve conflicts.  After this
//    stage, each value has only one layout associated with it.
//
// 4. Rewrite the IR by walking the function in dominance order. Since we
//    assume the IR is structured we just need to process the regions in the
//    correct order. For each op, rewrite it using the layout decided by the
//    analysis phase.
#ifdef __FLAGTREE_RLC_ENHANCE__
//
// On top of this baseline we add four optional phases, each guarded by a pass
// option; with every option off the pass is identical to the original.
//
//   Phase 1a (enable-cost-based-resolution): resolve conflicts by estimated
//     convert cost instead of the fixed "blocked for load/store, mma
//     otherwise" heuristic.
//
//   Phase 1b (enable-backward-propagation): push an encoding preference
//     backward from single-use writeback converts into their producer chains.
//
//   Phase 2 (enable-small-component-solving): retag a small closed component to
//     one encoding when a loop-depth-weighted cost model shows it removes a
//     convert without degrading memory access.
//
//   Phase 3 (enable-store-layout-rematerialization): drop a writeback convert
//     from an MMA value by rematerializing its address/mask chains in the
//     value's layout (see LayoutRematerialization).
//
// Phase 1b and Phase 2 require Phase 1a.
#endif // __FLAGTREE_RLC_ENHANCE__
class LayoutPropagation {
public:
  // Structure to keep track of the layout associated to a value.
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() {}
    llvm::SmallSetVector<Attribute, 8> encodings;
  };
#ifdef __FLAGTREE_RLC_ENHANCE__
  LayoutPropagation(FuncOp F, bool costBased = true, bool backwardProp = true,
                    bool smallComponentSolving = true,
                    ModuleAxisInfoAnalysis *axisInfo = nullptr)
      : funcOp(F), enableCostBasedResolution(costBased),
        enableBackwardPropagation(backwardProp),
        enableSmallComponentSolving(smallComponentSolving),
        axisInfoAnalysis(axisInfo) {}
#else  // __FLAGTREE_RLC_ENHANCE__
  LayoutPropagation(FuncOp F) : funcOp(F) {}
#endif // __FLAGTREE_RLC_ENHANCE__
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  // Add layouts given in `Info` to the uses of `value`.
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
#ifdef __FLAGTREE_RLC_ENHANCE__
  // Propagate layout preferences from convert_layout users to producers.
  bool propagateLayoutBackward();
  SmallVector<Value> propagateToOperands(Value value, LayoutInfo &info);
  // Add candidates for closed local components.
  bool solveSmallComponents();
#endif // __FLAGTREE_RLC_ENHANCE__
  // Set the encoding to all the values and fill out the values with new layout
  // in `changed`.
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  // Rewrite the IR for the full module.
  void rewrite();
  // Rewrite the IR for a region.
  void rewriteRegion(Region &R);
  // Rewrite an op based on the layout picked by the analysis.
  Operation *rewriteOp(Operation *op);
  // Rewrite a for op based on the layout picked by the analysis.
  Operation *rewriteForOp(scf::ForOp forOp);
  Operation *rewriteWhileOp(scf::WhileOp whileOp);
  Operation *rewriteIfOp(scf::IfOp ifOp);
  void rewriteYieldOp(scf::YieldOp yieldOp);
  void rewriteConditionOp(scf::ConditionOp conditionOp);
  void rewriteReduceToScalar(Operation *reduceOp);
  void rewriteAssertOp(AssertOp assertOp);
  Operation *cloneElementwise(OpBuilder &rewriter, Operation *op,
                              Attribute encoding);
  // Map the original value to the rewritten one.
  void map(Value old, Value newV);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  // Return the original value mapped to the new desired encoding.
  Value getRewrittenValue(Value value);
  // Dump the current stage of layout information.
  void dump();

private:
#ifdef __FLAGTREE_RLC_ENHANCE__
  bool hasLayoutPropagationExtensions() const;
  LayoutInfo *getOrCreateBackwardLayout(Value value);
  bool addBackwardEncoding(Value target, Attribute encoding);
  bool isCompatibleWithEncoding(Value value, Attribute encoding) const;
  int64_t getValueConversionCost(Value value, Attribute encoding) const;
  int64_t getDefiningOpConversionCost(Value value, Attribute encoding) const;
  int64_t getUseConversionCost(Value value, OpOperand &use, Attribute encoding,
                               int64_t cvtUnitCost) const;
  bool shouldUseCostBasedResolution(const LayoutInfo &info) const;
  bool isExtensionTouchedValue(Value value) const;
  bool addSmallComponentEncoding(Value target, Attribute encoding);
  using Proposal = llvm::MapVector<Value, Attribute>;
  bool addProposalValue(Proposal &proposal, Value target,
                        Attribute encoding) const;
  bool collectProducerClosure(Proposal &proposal, Value target,
                              Attribute encoding, int depth,
                              bool stopAtConvertSrc = false,
                              bool allowBlockArgs = false) const;
  Attribute getProposalEncoding(const Proposal &proposal, Value value) const;
  bool shouldAcceptProposal(const Proposal &proposal) const;
  int countConvertsRemovedByProposal(const Proposal &proposal) const;
  bool proposalAmplifiesScatterStore(const Proposal &proposal) const;
  bool proposalDecoalescesContiguousStore(const Proposal &proposal) const;
  bool proposalTouchesReductionOrScan(const Proposal &proposal) const;
  bool proposalDirectlyTouchesReductionOrScan(const Proposal &proposal) const;
  bool proposalHasOnlyPairedForBlockArgs(const Proposal &proposal) const;
  bool proposalHasRiskyScalarSideOutput(const Proposal &proposal) const;
  bool proposalHasRankChangingLayoutBridge(const Proposal &proposal) const;
  bool commitProposal(const Proposal &proposal);
#endif // __FLAGTREE_RLC_ENHANCE__
  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  // map of the values rewrite based on their encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rewriteMapping;
  SetVector<Operation *> opToDelete;
  FuncOp funcOp;
#ifdef __FLAGTREE_RLC_ENHANCE__
  bool enableCostBasedResolution;
  bool enableBackwardPropagation;
  bool enableSmallComponentSolving;
  // Address-pattern analysis for the phase extensions (null when disabled).
  ModuleAxisInfoAnalysis *axisInfoAnalysis;
  // Values first tracked by backward propagation.
  DenseSet<Value> backwardCreatedValues;
  // Values touched by backward propagation. Their later forward propagation is
  // restricted to local chains.
  DenseSet<Value> backwardTouchedValues;
  // Encoding selected by a closed component proposal.
  DenseMap<Value, Attribute> smallComponentPreferredEncoding;
#endif // __FLAGTREE_RLC_ENHANCE__
};

class LayoutRematerialization {
public:
#ifdef __FLAGTREE_RLC_ENHANCE__
  LayoutRematerialization(FuncOp F, bool duplicableLoadRemat = false,
                          ModuleAxisInfoAnalysis *axisInfo = nullptr,
                          bool storeLayoutRemat = false)
      : funcOp(F), enableDuplicableLoadRemat(duplicableLoadRemat),
        enableStoreLayoutRemat(storeLayoutRemat), axisInfoAnalysis(axisInfo) {}
#else  // __FLAGTREE_RLC_ENHANCE__
  LayoutRematerialization(FuncOp F) : funcOp(F) {}
#endif // __FLAGTREE_RLC_ENHANCE__

  // Map the original value to the remat'ed one.
  void addRematValue(Value old, Attribute encoding, Value newV);
  // Get the remat'ed value in the given encoding, if one already exists and
  // is different then the layout conversion root.
  Value getRematValue(Value value, Attribute encoding) const {
    return rematMapping.lookup({value, encoding});
  }

  void cleanup();
  bool backwardRematerialization();
  void backwardRematerialization(ConvertLayoutOp convertOp);
#ifdef __FLAGTREE_RLC_ENHANCE__
  // Phase 3: eliminate writeback convert_layout ops by rematerializing the
  // writeback's address/mask chains in the written value's layout. A writeback
  // is a tt.store or a tt.atomic_rmw whose result is unused. Returns true if
  // any writeback was rewritten.
  bool rematerializeStoreLayout();
  bool rematerializeWritebackLayout(Operation *writebackOp);
  bool rematerializeLocalStoreLayout();
  bool rematerializeLocalStoreLayout(LocalStoreOp storeOp);
#endif // __FLAGTREE_RLC_ENHANCE__
  // TODO: Merge the three hoistConvert*(); functions as they are duplicate code
  void hoistConvertDotOperand();
  void hoistConvertDotOperand(ConvertLayoutOp convertOp);
  void hoistConvertOnTopOfExtOrBroadcast();
  void hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp convertOp);
  void hoistConvertIntoConditionals();
  void hoistConvertIntoConditionals(ConvertLayoutOp convertOp);
  void rewriteSlice(SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
                    ConvertLayoutOp convertOp, IRMapping &mapping);
  void rewriteSlice(SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
                    ConvertLayoutOp convertOp);

  LogicalResult
  getConvertBackwardSlice(OpOperand &root, Attribute rootEncoding,
                          SetVector<Value> &slice,
                          DenseMap<Value, Attribute> &layout,
                          std::function<bool(Operation *)> stopPropagation);

  LogicalResult getRematerializableSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
#ifdef __FLAGTREE_RLC_ENHANCE__
      std::function<bool(Operation *)> stopPropagation = nullptr,
      bool allowDuplicableLoads = false);
#else  // __FLAGTREE_RLC_ENHANCE__
      std::function<bool(Operation *)> stopPropagation = nullptr);
#endif // __FLAGTREE_RLC_ENHANCE__

private:
  void updateRematMapping(SmallVector<std::tuple<Value, Value>> &values);
  // Existing tuples of (value, layout) that needs to be updated when recreating
  // scf ops. This prevents keeping track of Values that have been delete when
  // rewriting slices.
  DenseMap<Value, Attribute> mappedValues;
  // map of the values remat based on encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rematMapping;
  // DenseMap<std::pair<Operation*, Attribute>, Operation*>
  SetVector<Operation *> opToDelete;
  FuncOp funcOp;
#ifdef __FLAGTREE_RLC_ENHANCE__
  // Phase 3: allow rematerialization slices to duplicate small vector loads
  // whose access pattern is preserved in the target encoding (see
  // isDuplicableLoadInEncoding).
  bool enableDuplicableLoadRemat;
  // Phase 3: this instance runs writeback-store rematerialization, whose
  // rewrites can leave a still-used op in opToDelete. Enables the cleanup()
  // live-use guard; false on the baseline (and duplicable-load-only) paths.
  bool enableStoreLayoutRemat;
  // Address-pattern analysis for the duplicable-load gate (null when the
  // phase is disabled).
  ModuleAxisInfoAnalysis *axisInfoAnalysis;
#endif // __FLAGTREE_RLC_ENHANCE__
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

void LayoutRematerialization::addRematValue(Value old, Attribute encoding,
                                            Value newV) {
  LDBG("addRematValue " << old << " encoding " << encoding << " " << newV);
  rematMapping[{old, encoding}] = newV;
  mappedValues[old] = encoding;
}

// Remove unneeded values now that we are done with the rematMapping.
void LayoutRematerialization::cleanup() {
#ifdef __FLAGTREE_RLC_ENHANCE__
  // Phase 3 writeback rematerialization can queue an op that still has live
  // uses (e.g. a convert shared with another writeback), which must not be
  // erased. Without Phase 3 the queue only holds fully-replaced ops, so we keep
  // the original unconditional erase to stay byte-for-byte identical.
  bool guardLiveUses = enableDuplicableLoadRemat || enableStoreLayoutRemat;
  for (Operation *op : llvm::reverse(opToDelete)) {
    if (guardLiveUses && !allResultsHaveNoUses(op))
      continue;
    op->erase();
  }
#else  // __FLAGTREE_RLC_ENHANCE__
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
#endif // __FLAGTREE_RLC_ENHANCE__
}

#ifdef __FLAGTREE_RLC_ENHANCE__
// Forward declaration (defined further below in the rematerialization section).
static int64_t getByteCount(Value result, int64_t minElementCount,
                            int64_t minBitWidth);
#endif // __FLAGTREE_RLC_ENHANCE__

// Return true if the op is an op with a layout we don't want to change. We will
// propagate the layout starting from anchor ops.
bool isLayoutAnchor(Operation *op) {
  if (isa<DescriptorOpInterface>(op))
    return true;
  if (isa<LoadOp, StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<DotOp, DotScaledOp, nvidia_gpu::WarpGroupDotOp, AtomicRMWOp,
          AtomicCASOp, triton::nvidia_gpu::TMEMLoadOp>(op))
    return true;
  if (auto gatherOp = dyn_cast<GatherOp>(op))
    return gatherOp.getEfficientLayout();

  // Heuristic: Mark permuting reshape as a layout anchor.  Its dst can be
  // anything, so it stops forward-propagation of layouts.  We rely on the
  // backwards pass to fix it up if necessary.  (If we didn't do this, then
  // anything following the reshape won't be covered by the forward pass at
  // all.)
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return reshape.getAllowReorder();

  return false;
}

#ifdef __FLAGTREE_RLC_ENHANCE__
bool LayoutPropagation::hasLayoutPropagationExtensions() const {
  return enableCostBasedResolution || enableBackwardPropagation ||
         enableSmallComponentSolving;
}
#endif // __FLAGTREE_RLC_ENHANCE__

void LayoutPropagation::initAnchorLayout() {
  auto addAnchor = [&](Value v) {
    if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
#ifdef __FLAGTREE_RLC_ENHANCE__
      if (!hasLayoutPropagationExtensions() || tensorType.getEncoding())
        layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
#else  // __FLAGTREE_RLC_ENHANCE__
      layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
#endif // __FLAGTREE_RLC_ENHANCE__
    }
  };

  // Consider function args as anchors.  This makes it easier to write tests --
  // you can pass a tensor with an encoding as an arg, instead of explicitly
  // calling tt.load.
  for (auto arg : funcOp.getArguments()) {
    addAnchor(arg);
  }

  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        addAnchor(result);
      }
    }
  });
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool hasChanged = false;
    for (auto encoding : info.encodings) {
      Attribute dstEncoding;
      if (isa<ConvertLayoutOp>(op)) {
        // Try to remove the convert by making the dst encoding match the source
        // encoding.
        dstEncoding = encoding;
      } else {
        dstEncoding = inferDstEncoding(op, encoding);
      }
      if (dstEncoding)
        hasChanged |= layouts[value].encodings.insert(dstEncoding);
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

#ifdef __FLAGTREE_RLC_ENHANCE__
static bool isBackwardPropagationForwardUser(Operation *op);
static void
collectUseTargets(OpOperand &use, Attribute enc,
                  SmallVectorImpl<std::pair<Value, Attribute>> &targets);
#endif // __FLAGTREE_RLC_ENHANCE__

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
#ifdef __FLAGTREE_RLC_ENHANCE__
  bool restrictToBackwardLocalChain = backwardTouchedValues.contains(value);
#endif // __FLAGTREE_RLC_ENHANCE__
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
#ifdef __FLAGTREE_RLC_ENHANCE__
    if (restrictToBackwardLocalChain && !isBackwardPropagationForwardUser(user))
      continue;
#endif // __FLAGTREE_RLC_ENHANCE__
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      setEncoding({arg}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate;
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        valuesToPropagate.push_back(parent->getResult(use.getOperandNumber()));
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
        valuesToPropagate.push_back(
            whileOp.getBeforeArguments()[use.getOperandNumber()]);
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        setEncoding(valuesToPropagate, info, changed, user);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip arg 0 as it is the condition.
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      setEncoding({afterArg, result}, info, changed, user);
      continue;
    }
    if (auto dotWaitOp = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
      unsigned opIndex = use.getOperandNumber();
      Value result = dotWaitOp->getResult(opIndex);
      setEncoding(result, info, changed, user);
      continue;
    }
    if (auto gatherOp = dyn_cast<GatherOp>(user)) {
      // Propagate the layout through the indices only, and if the layout does
      // not have an efficient layout set.
      if (!gatherOp.getEfficientLayout() &&
          &use == &gatherOp.getIndicesMutable()) {
        setEncoding(gatherOp.getResult(), info, changed, user);
        continue;
      }
    }
    if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        user->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            ConvertLayoutOp>(user)) {
      setEncoding(user->getResults(), info, changed, user);
      continue;
    }
  }
#ifdef __FLAGTREE_RLC_ENHANCE__
  if (restrictToBackwardLocalChain)
    backwardTouchedValues.insert(changed.begin(), changed.end());
#endif // __FLAGTREE_RLC_ENHANCE__
  return changed;
}

void LayoutPropagation::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);

    LLVM_DEBUG({
      DBGS() << "propagateLayout considering " << currentValue << ", which has "
             << info.encodings.size() << " candidate encoding(s):\n";
      for (Attribute encoding : info.encodings)
        DBGS() << "  " << encoding << "\n";
      DBGS() << "changed: " << changed.size() << "\n";
    });

    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

#ifdef __FLAGTREE_RLC_ENHANCE__
static bool hasSingleUse(Value value) {
  unsigned numUses = 0;
  for (OpOperand &use : value.getUses()) {
    (void)use;
    if (++numUses > 1)
      return false;
  }
  return numUses == 1;
}

static bool isFrozenLayoutValue(Value value) {
  Operation *defOp = value.getDefiningOp();
  return !defOp || isLayoutAnchor(defOp);
}

static bool isInsideStructuredControlFlow(Operation *op) {
  return op->getParentOfType<scf::ForOp>() ||
         op->getParentOfType<scf::WhileOp>() ||
         op->getParentOfType<scf::IfOp>();
}

static bool canCreateBackwardLayout(Value value);
static bool isSmallComponentForbiddenOp(Operation *op);
static bool isSmallComponentRetaggable(Value value);
static bool isExpensiveMathOp(Operation *op);
static bool isSliceEncoding(Attribute encoding);
static bool isEncodingUniformPureOp(Operation *op);
static bool preservesWritebackMemoryAccess(RankedTensorType ptrType,
                                           RankedTensorType valueType,
                                           Attribute targetEncoding,
                                           bool allowNarrowerContiguity);
static unsigned getContigAlongMemoryOrder(RankedTensorType type);
static bool isOrderIndifferentAccess(ModuleAxisInfoAnalysis *axisInfo,
                                     Value ptr);
static bool convertResultReachesAnotherConvert(ConvertLayoutOp convertOp);

// Seed backward propagation onto `value`; only non-frozen local producers with
// a blocked encoding qualify. A fresh entry is primed with the current encoding
// and recorded in backwardCreatedValues so the later forward pass treats it as
// a local candidate rather than an anchor.
LayoutPropagation::LayoutInfo *
LayoutPropagation::getOrCreateBackwardLayout(Value value) {
  Attribute encoding = getTensorEncoding(value);
  if (!isa_and_nonnull<BlockedEncodingAttr>(encoding) ||
      !canCreateBackwardLayout(value))
    return nullptr;

  auto &entry = layouts[value];
  if (entry.encodings.empty()) {
    entry.encodings.insert(encoding);
    backwardCreatedValues.insert(value);
  }
  return &entry;
}

bool LayoutPropagation::addBackwardEncoding(Value target, Attribute encoding) {
  if (!encoding)
    return false;
  LayoutInfo *entry = getOrCreateBackwardLayout(target);
  if (!entry)
    return false;
  bool inserted = entry->encodings.insert(encoding);
  if (inserted)
    backwardTouchedValues.insert(target);
  return inserted;
}

bool LayoutPropagation::addSmallComponentEncoding(Value target,
                                                  Attribute encoding) {
  Attribute originalEncoding = getTensorEncoding(target);
  if (!encoding || !originalEncoding)
    return false;
  auto &entry = layouts[target];
  if (entry.encodings.empty())
    entry.encodings.insert(originalEncoding);
  bool inserted = entry.encodings.insert(encoding);
  smallComponentPreferredEncoding[target] = encoding;
  return inserted;
}

bool LayoutPropagation::addProposalValue(Proposal &proposal, Value target,
                                         Attribute encoding) const {
  if (!encoding || !getTensorEncoding(target))
    return false;
  auto [it, inserted] = proposal.insert({target, encoding});
  if (!inserted)
    return it->second == encoding;
  return true;
}

Attribute LayoutPropagation::getProposalEncoding(const Proposal &proposal,
                                                 Value value) const {
  if (auto it = proposal.find(value); it != proposal.end())
    return it->second;
  if (auto preferred = smallComponentPreferredEncoding.lookup(value))
    return preferred;
  auto layoutIt = layouts.find(value);
  if (layoutIt == layouts.end() || layoutIt->second.encodings.empty())
    return getTensorEncoding(value);
  const LayoutInfo &info = layoutIt->second;
  if (info.encodings.size() == 1)
    return *info.encodings.begin();
  // Predict the resolveConflicts outcome for a multi-candidate value no
  // proposal touches: mirror the legacy preference (blocked for memory ops, MMA
  // otherwise) rather than the worklist-dependent first candidate, so the cost
  // model sees the convert that will materialize on the real boundary.
  Operation *defOp = value.getDefiningOp();
  bool isLoadOrStore =
      defOp && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(defOp);
  for (Attribute encoding : info.encodings) {
    if ((isLoadOrStore && isa<BlockedEncodingAttr>(encoding)) ||
        (!isLoadOrStore && isa<MmaEncodingTrait>(encoding)))
      return encoding;
  }
  return *info.encodings.begin();
}

// Recursively gather `target`'s producer chain into `proposal`, all retagged to
// `encoding`, recursing through encoding-transferring producers (elementwise,
// broadcast, expand_dims, reshape, single-input scan, loop-carried values) and
// pairing loop-carried values with their result/init/yield. Returns false when
// any producer cannot be safely retagged (anchor, forbidden op, failed source
// inference, access-pattern-breaking load, depth exhaustion).
bool LayoutPropagation::collectProducerClosure(Proposal &proposal, Value target,
                                               Attribute encoding, int depth,
                                               bool stopAtConvertSrc,
                                               bool allowBlockArgs) const {
  if (depth <= 0)
    return false;
  // A value already in the proposal has its producer chain collected (or in
  // collection by an enclosing frame). This both deduplicates shared chains
  // and terminates loop-carried cycles (iter arg -> yield -> iter arg).
  if (auto it = proposal.find(target); it != proposal.end())
    return it->second == encoding;
  if (!addProposalValue(proposal, target, encoding))
    return false;

  // A loop-carried value is retagged together with its paired loop result,
  // init value and yielded value so the loop signature stays consistent.
  auto collectLoopCarried = [&](scf::ForOp forOp, unsigned resultNumber) {
    Value regionArg = forOp.getRegionIterArg(resultNumber);
    Value result = forOp.getResult(resultNumber);
    if (!addProposalValue(proposal, regionArg, encoding) ||
        !addProposalValue(proposal, result, encoding))
      return false;
    if (!collectProducerClosure(proposal, forOp.getInitArgs()[resultNumber],
                                encoding, depth - 1, stopAtConvertSrc,
                                allowBlockArgs))
      return false;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    return collectProducerClosure(proposal, yieldOp.getOperand(resultNumber),
                                  encoding, depth - 1, stopAtConvertSrc,
                                  /*allowBlockArgs=*/true);
  };

  Operation *defOp = target.getDefiningOp();
  if (!defOp) {
    auto blockArg = cast<BlockArgument>(target);
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      unsigned numIndVars = forOp.getNumInductionVars();
      if (blockArg.getArgNumber() >= numIndVars)
        return collectLoopCarried(forOp, blockArg.getArgNumber() - numIndVars);
    }
    return allowBlockArgs;
  }
  // A single-input scan preserves the encoding (tt.scan is
  // SameOperandsAndResultEncoding, its combine region only sees scalars), so it
  // can be retagged with its input chain. Multi-operand scans stay anchors.
  if (auto scanOp = dyn_cast<ScanOp>(defOp)) {
    if (scanOp.getSrcs().size() != 1)
      return false;
    return collectProducerClosure(proposal, scanOp.getSrcs()[0], encoding,
                                  depth - 1, stopAtConvertSrc, allowBlockArgs);
  }
  if (isSmallComponentForbiddenOp(defOp))
    return false;
  if (isa<StoreOp>(defOp))
    return false;
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    auto result = dyn_cast<OpResult>(target);
    if (!result || result.getOwner() != forOp)
      return false;
    return collectLoopCarried(forOp, result.getResultNumber());
  }

  if (isa<scf::WhileOp>(defOp))
    return false;

  // A conditionally produced value is retagged together with both branch
  // yields so the scf.if signature stays consistent.
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    unsigned resultNumber = cast<OpResult>(target).getResultNumber();
    return collectProducerClosure(
               proposal, ifOp.thenYield().getOperand(resultNumber), encoding,
               depth - 1, stopAtConvertSrc, allowBlockArgs) &&
           collectProducerClosure(
               proposal, ifOp.elseYield().getOperand(resultNumber), encoding,
               depth - 1, stopAtConvertSrc, allowBlockArgs);
  }

  if (auto loadOp = dyn_cast<LoadOp>(defOp)) {
    // Retag an expensive (coalesced) load only when the new layout keeps its
    // global access pattern (same fastest dimension, no lost contiguity). An
    // access with no contiguity to preserve may reorder freely, but only
    // outside loops: reshaping a loop-resident scatter stream degrades it.
    if (isExpensiveLoadOrStore(defOp)) {
      auto ptrType = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
      auto resultType = dyn_cast<RankedTensorType>(loadOp.getType());
      bool inLoop = defOp->getParentOfType<scf::ForOp>() != nullptr ||
                    defOp->getParentOfType<scf::WhileOp>() != nullptr;
      if (!preservesWritebackMemoryAccess(ptrType, resultType, encoding,
                                          /*allowNarrowerContiguity=*/false) &&
          (inLoop ||
           !isOrderIndifferentAccess(axisInfoAnalysis, loadOp.getPtr())))
        return false;
    }
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    for (OpOperand &operand : loadOp->getOpOperands()) {
      Value operandValue = operand.get();
      if (!isa<RankedTensorType>(operandValue.getType()))
        continue;
      if (operandValue.getDefiningOp() &&
          isLayoutAnchor(operandValue.getDefiningOp()))
        return false;
      if (!collectProducerClosure(proposal, operandValue, srcEncoding,
                                  depth - 1, stopAtConvertSrc, allowBlockArgs))
        return false;
    }
    return true;
  }

  if (auto expandOp = dyn_cast<ExpandDimsOp>(defOp)) {
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    return collectProducerClosure(proposal, expandOp.getSrc(), srcEncoding,
                                  depth - 1, stopAtConvertSrc, allowBlockArgs);
  }

  if (auto cvtOp = dyn_cast<ConvertLayoutOp>(defOp)) {
    if (stopAtConvertSrc)
      return true;
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    return collectProducerClosure(proposal, cvtOp.getSrc(), srcEncoding,
                                  depth - 1, stopAtConvertSrc, allowBlockArgs);
  }

  if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    return collectProducerClosure(proposal, broadcastOp.getSrc(), srcEncoding,
                                  depth - 1, stopAtConvertSrc, allowBlockArgs);
  }

  if (defOp->hasTrait<OpTrait::Elementwise>() ||
      defOp->hasTrait<OpTrait::SameOperandsAndResultEncoding>()) {
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    for (Value operand : defOp->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (!collectProducerClosure(proposal, operand, srcEncoding, depth - 1,
                                  stopAtConvertSrc, allowBlockArgs))
        return false;
    }
    return true;
  }

  if (auto reshapeOp = dyn_cast<ReshapeOp>(defOp)) {
    if (!reshapeOp.getAllowReorder() || reshapeOp.getEfficientLayout())
      return false;
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    return collectProducerClosure(proposal, reshapeOp.getSrc(), srcEncoding,
                                  depth - 1, stopAtConvertSrc, allowBlockArgs);
  }

  if (auto gatherOp = dyn_cast<GatherOp>(defOp)) {
    if (gatherOp.getEfficientLayout())
      return false;
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return false;
    if (!collectProducerClosure(proposal, gatherOp.getSrc(), srcEncoding,
                                depth - 1, stopAtConvertSrc, allowBlockArgs))
      return false;
    if (!collectProducerClosure(proposal, gatherOp.getIndices(), encoding,
                                depth - 1, stopAtConvertSrc, allowBlockArgs))
      return false;
    return true;
  }

  // Encoding-uniform pure ops transfer the layout unchanged onto all their
  // tensor operands.
  if (isEncodingUniformPureOp(defOp)) {
    for (Value operand : defOp->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (!collectProducerClosure(proposal, operand, encoding, depth - 1,
                                  stopAtConvertSrc, allowBlockArgs))
        return false;
    }
    return true;
  }

  if (canFoldIntoConversion(defOp, encoding))
    return true;

  return false;
}

// Execution-frequency weight for a convert of `value`, by the loop depth of its
// definition (converts are inserted next to the definition). This makes the
// cost model favor removing hot in-loop converts over one-shot ones.
static int64_t getLoopFrequencyWeight(Value value) {
  constexpr int64_t kLoopIterationWeight = 8;
  constexpr int kMaxWeightedLoopDepth = 3;
  // For an op result the convert lands in the same region as the defining op;
  // for a loop-carried block argument it lands inside the loop itself.
  Operation *enclosing;
  if (Operation *defOp = value.getDefiningOp())
    enclosing = defOp->getParentOp();
  else
    enclosing = cast<BlockArgument>(value).getOwner()->getParentOp();

  int depth = 0;
  for (Operation *op = enclosing; op && depth < kMaxWeightedLoopDepth;
       op = op->getParentOp())
    if (isa<scf::ForOp, scf::WhileOp>(op))
      ++depth;

  int64_t weight = 1;
  for (int i = 0; i < depth; ++i)
    weight *= kLoopIterationWeight;
  return weight;
}

// Cost-model gate for a Phase 2 proposal: sum the convert cost across every
// proposal value's uses and defining-op operands under the current encodings
// versus the proposed ones (each convert weighted by its loop depth), and
// accept only when the proposal strictly lowers the total.
bool LayoutPropagation::shouldAcceptProposal(const Proposal &proposal) const {
  int64_t beforeCost = 0;
  int64_t afterCost = 0;

  auto getOriginalEncoding = [&](Value value) -> Attribute {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return {};
    return tensorType.getEncoding();
  };

  auto getCurrentEncoding = [&](Value value) -> Attribute {
    if (!isa<RankedTensorType>(value.getType()))
      return {};
    Proposal emptyProposal;
    return getProposalEncoding(emptyProposal, value);
  };

  auto getSelectedEncoding = [&](Value value) -> Attribute {
    if (!isa<RankedTensorType>(value.getType()))
      return {};
    return getProposalEncoding(proposal, value);
  };

  auto getCvtCost = [](Value value) -> int64_t {
    int64_t cost = 32 * getByteCount(value, 32, 32);
    return (cost == 0 ? 1 : cost) * getLoopFrequencyWeight(value);
  };

  auto countUseCvtCost = [&](Value value, Attribute encoding,
                             bool useProposal) {
    if (!isa<RankedTensorType>(value.getType()))
      return int64_t(0);
    int64_t cost = 0;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      SmallVector<std::pair<Value, Attribute>> targets;
      collectUseTargets(use, encoding, targets);
      if (targets.empty() &&
          (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
           user->hasTrait<OpTrait::SameLoadStoreOperandsEncoding>())) {
        if (useProposal &&
            user->hasTrait<OpTrait::SameLoadStoreOperandsEncoding>()) {
          bool compatible = true;
          for (Value operand : user->getOperands()) {
            if (!isa<RankedTensorType>(operand.getType()))
              continue;
            Attribute operandEncoding = getSelectedEncoding(operand);
            if (!operandEncoding || operandEncoding != encoding) {
              compatible = false;
              break;
            }
          }
          if (compatible)
            continue;
        }
        auto useType = dyn_cast<RankedTensorType>(value.getType());
        if (useType && useType.getEncoding() != encoding)
          cost += getCvtCost(value);
        continue;
      }
      for (auto &[targetVal, dstEnc] : targets) {
        Attribute targetEncoding = useProposal ? getSelectedEncoding(targetVal)
                                               : getCurrentEncoding(targetVal);
        if (!targetEncoding || targetEncoding != dstEnc)
          cost += getCvtCost(value);
      }
    }
    return cost;
  };

  auto countDefOperandCvtCost = [&](Value value, Attribute encoding,
                                    bool useProposal) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return int64_t(0);
    if (isa<ConvertLayoutOp>(defOp)) {
      Value src = defOp->getOperand(0);
      Attribute srcEncoding =
          useProposal ? getSelectedEncoding(src) : getCurrentEncoding(src);
      if (!srcEncoding || srcEncoding != encoding)
        return getCvtCost(src);
      return int64_t(0);
    }
    if (!isa<ReduceOp, scf::ForOp, scf::WhileOp, scf::IfOp>(defOp))
      return int64_t(0);
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      return int64_t(0);

    int64_t cost = 0;
    for (Value operand : defOp->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      Attribute operandEncoding = useProposal ? getSelectedEncoding(operand)
                                              : getCurrentEncoding(operand);
      if (!operandEncoding || operandEncoding != srcEncoding)
        cost += getCvtCost(operand);
    }
    return cost;
  };

  for (auto &it : proposal) {
    Value value = it.first;
    Attribute encoding = it.second;
    Attribute currentEncoding = getCurrentEncoding(value);
    if (!currentEncoding)
      currentEncoding = getOriginalEncoding(value);
    beforeCost += countUseCvtCost(value, currentEncoding,
                                  /*useProposal=*/false);
    beforeCost += countDefOperandCvtCost(value, currentEncoding,
                                         /*useProposal=*/false);
    afterCost += countUseCvtCost(value, encoding, /*useProposal=*/true);
    afterCost += countDefOperandCvtCost(value, encoding,
                                        /*useProposal=*/true);
  }

  return afterCost < beforeCost;
}

// Count the convert_layout ops a proposal collapses: a convert whose source and
// result encodings become equal under the proposal but differed originally.
int LayoutPropagation::countConvertsRemovedByProposal(
    const Proposal &proposal) const {
  int removed = 0;
  for (auto &it : proposal) {
    Value value = it.first;
    auto cvtOp = dyn_cast_or_null<ConvertLayoutOp>(value.getDefiningOp());
    if (!cvtOp)
      continue;

    Attribute originalSrcEncoding = getTensorEncoding(cvtOp.getSrc());
    Attribute originalResultEncoding = getTensorEncoding(cvtOp.getResult());
    if (!originalSrcEncoding || !originalResultEncoding)
      continue;

    Attribute srcEncoding = getProposalEncoding(proposal, cvtOp.getSrc());
    Attribute resultEncoding = getProposalEncoding(proposal, cvtOp.getResult());
    if (!srcEncoding)
      srcEncoding = originalSrcEncoding;
    if (!resultEncoding)
      resultEncoding = originalResultEncoding;
    if (srcEncoding == resultEncoding &&
        originalSrcEncoding != originalResultEncoding)
      ++removed;
  }
  return removed;
}

// True if `value` is, within `depth` producer steps, just a copy of a coalesced
// ("expensive") load: only converts, casts and cheap layout-preserving
// elementwise ops ending at such a load, with no compute-bound work
// (transcendental math or lane-mixing ops). Constants/splats/make_range are
// neutral fan-in. The depth bound excludes deep reduction/scan networks.
static bool isCoalescedLoadCopyValue(Value value, int depth) {
  if (depth <= 0 || !isa<RankedTensorType>(value.getType()))
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<LoadOp>(def))
    return isExpensiveLoadOrStore(def);
  if (isa<ConvertLayoutOp>(def))
    return isCoalescedLoadCopyValue(def->getOperand(0), depth - 1);
  bool isCheapCopyOp =
      (def->hasTrait<OpTrait::Elementwise>() || isa<BitcastOp>(def)) &&
      def->getNumRegions() == 0 && !isExpensiveMathOp(def);
  if (!isCheapCopyOp)
    return false;
  bool reachedLoad = false;
  for (Value operand : def->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue; // Scalar fan-in (e.g. a subtraction constant) is neutral.
    Operation *operandDef = operand.getDefiningOp();
    if (operandDef && isa<arith::ConstantOp, SplatOp, MakeRangeOp>(operandDef))
      continue; // Broadcast of a scalar/iota is neutral, not real compute.
    if (!isCoalescedLoadCopyValue(operand, depth - 1))
      return false;
    reachedLoad = true;
  }
  return reachedLoad;
}

// True if the function performs heavy on-chip compute (reduction, scan, atomic
// or matmul). Such kernels are compute-bound, so collapsing converts on a
// scatter store still pays off even when the store widens; a pure data-movement
// kernel has no compute to amortize an amplified store against.
static bool functionHasComputeAnchors(FuncOp funcOp) {
  bool hasCompute = false;
  funcOp.walk([&](Operation *op) {
    if (isa<ReduceOp, ScanOp, AtomicRMWOp, AtomicCASOp, DotOp, DotScaledOp,
            nvidia_gpu::WarpGroupDotOp>(op))
      hasCompute = true;
  });
  return hasCompute;
}

// True if the proposal reaches at least one global store and every such store
// writes only a coalesced-load copy (see isCoalescedLoadCopyValue). This marks
// the memory-bound Load/Store passthrough, where amplifying a scatter store to
// remove its bridging convert is a net loss.
static bool proposalStoresOnlyLoadCopies(
    const llvm::MapVector<Value, Attribute> &proposal) {
  constexpr int kMaxLoadCopyDepth = 4;
  bool sawStore = false;
  for (auto &it : proposal) {
    for (Operation *user : it.first.getUsers()) {
      auto storeOp = dyn_cast<StoreOp>(user);
      if (!storeOp)
        continue;
      sawStore = true;
      if (!isCoalescedLoadCopyValue(storeOp.getValue(), kMaxLoadCopyDepth))
        return false;
    }
  }
  return sawStore;
}

// A store retag that raises per-thread contiguity beyond what the real address
// pattern supports does not widen the store; it turns each thread's single
// scattered access into sizePerThread scattered ones. Only worth committing
// when it collapses more than one convert (enforced by the caller).
bool LayoutPropagation::proposalAmplifiesScatterStore(
    const Proposal &proposal) const {
  for (auto &it : proposal) {
    for (Operation *user : it.first.getUsers()) {
      auto storeOp = dyn_cast<StoreOp>(user);
      if (!storeOp)
        continue;
      auto ptrType = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());
      if (!ptrType || !ptrType.getEncoding())
        continue;
      Attribute targetEncoding =
          getProposalEncoding(proposal, storeOp.getPtr());
      if (!targetEncoding || targetEncoding == ptrType.getEncoding())
        continue;
      unsigned targetContig =
          getContigAlongMemoryOrder(ptrType.cloneWithEncoding(targetEncoding));
      unsigned currentContig = getContigAlongMemoryOrder(ptrType);
      if (targetContig <= currentContig)
        continue;
      // The raise is genuine only when the addresses really are contiguous
      // for at least the new per-thread width.
      int64_t realContig = 1;
      if (axisInfoAnalysis) {
        if (AxisInfo *info = axisInfoAnalysis->getAxisInfo(storeOp.getPtr())) {
          SmallVector<unsigned> order = getOrderForMemory(ptrType);
          if (!order.empty())
            realContig = std::min<int64_t>(info->getContiguity(order.front()),
                                           ptrType.getDimSize(order.front()));
        }
      }
      if (int64_t(targetContig) > realContig)
        return true;
    }
  }
  return false;
}

// Complement of proposalAmplifiesScatterStore: reject a retag that widens a
// contiguous (coalesced) store past what its alignment can vectorize, so the
// extra per-thread elements stay scalar and stride the warp apart. A genuine
// scatter has no coalescing to lose and is left alone.
bool LayoutPropagation::proposalDecoalescesContiguousStore(
    const Proposal &proposal) const {
  if (!axisInfoAnalysis)
    return false;
  for (auto &it : proposal) {
    for (Operation *user : it.first.getUsers()) {
      auto storeOp = dyn_cast<StoreOp>(user);
      if (!storeOp)
        continue;
      auto ptrType = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());
      if (!ptrType || !ptrType.getEncoding())
        continue;
      Attribute targetEncoding =
          getProposalEncoding(proposal, storeOp.getPtr());
      if (!targetEncoding || targetEncoding == ptrType.getEncoding())
        continue;
      unsigned targetContig =
          getContigAlongMemoryOrder(ptrType.cloneWithEncoding(targetEncoding));
      unsigned currentContig = getContigAlongMemoryOrder(ptrType);
      if (targetContig <= currentContig)
        continue;
      SmallVector<unsigned> order = getOrderForMemory(ptrType);
      if (order.empty())
        continue;
      AxisInfo *info = axisInfoAnalysis->getAxisInfo(storeOp.getPtr());
      if (!info)
        continue;
      // Only a store contiguous past the current width was coalesced; a scatter
      // has nothing to lose.
      int64_t rawContig = std::min<int64_t>(info->getContiguity(order.front()),
                                            ptrType.getDimSize(order.front()));
      if (rawContig <= currentContig)
        continue;
      // If the alignment backs the wider width the store still vectorizes; only
      // a width past the alignment decoalesces it.
      unsigned align = axisInfoAnalysis->getAlignment(storeOp.getPtr());
      if (int64_t(targetContig) > int64_t(align))
        return true;
    }
  }
  return false;
}

// A scan whose tensor operands and results are all retagged together by the
// proposal keeps a consistent layout (tt.scan is encoding-preserving and its
// combine region only sees scalars); it does not anchor the component the way
// a reduction or a partially covered scan does.
static bool
isScanFullyInProposal(Operation *op,
                      const llvm::MapVector<Value, Attribute> &proposal) {
  auto scanOp = dyn_cast<ScanOp>(op);
  if (!scanOp)
    return false;
  auto covered = [&](Value value) {
    return !isa<RankedTensorType>(value.getType()) || proposal.count(value);
  };
  return llvm::all_of(scanOp->getOperands(), covered) &&
         llvm::all_of(scanOp->getResults(), covered);
}

bool LayoutPropagation::proposalTouchesReductionOrScan(
    const Proposal &proposal) const {
  auto isInProposal = [&](Value value) {
    return proposal.find(value) != proposal.end();
  };

  DenseSet<Value> visited;
  SmallVector<Value> queue;
  for (auto &it : proposal) {
    Value value = it.first;
    Operation *defOp = value.getDefiningOp();
    if (isa_and_nonnull<ReduceOp, ScanOp>(defOp) &&
        !isScanFullyInProposal(defOp, proposal))
      return true;
    queue.push_back(value);
  }

  while (!queue.empty()) {
    Value value = queue.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    for (Operation *user : value.getUsers()) {
      if (isa<ReduceOp, ScanOp>(user) && !isScanFullyInProposal(user, proposal))
        return true;
      if (isLayoutAnchor(user) ||
          isa<scf::ForOp, scf::WhileOp, scf::IfOp>(user))
        continue;
      if (!(user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
            user->hasTrait<OpTrait::Elementwise>() ||
            isa<ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
                ConvertLayoutOp>(user)))
        continue;
      for (Value result : user->getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        if (!isInProposal(result))
          queue.push_back(result);
      }
    }
  }

  return false;
}

bool LayoutPropagation::proposalDirectlyTouchesReductionOrScan(
    const Proposal &proposal) const {
  for (auto &it : proposal) {
    Value value = it.first;
    Operation *defOp = value.getDefiningOp();
    if (isa_and_nonnull<ReduceOp, ScanOp>(defOp) &&
        !isScanFullyInProposal(defOp, proposal))
      return true;
    for (Operation *user : value.getUsers())
      if (isa<ReduceOp, ScanOp>(user) && !isScanFullyInProposal(user, proposal))
        return true;
  }
  return false;
}

bool LayoutPropagation::proposalHasOnlyPairedForBlockArgs(
    const Proposal &proposal) const {
  for (auto &it : proposal) {
    auto blockArg = dyn_cast<BlockArgument>(it.first);
    if (!blockArg)
      continue;

    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return false;
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber == 0)
      return false;
    unsigned iterArgNumber = argNumber - 1;
    if (iterArgNumber >= forOp.getNumResults())
      return false;

    Value result = forOp.getResult(iterArgNumber);
    auto resultIt = proposal.find(result);
    if (resultIt == proposal.end() || resultIt->second != it.second)
      return false;
  }
  return true;
}

// Conservatively skip a rank-1 loop-carried accumulator chain that writes a
// 16-bit-float scalar side output. Its retag payoff hinges on backend
// scheduling that the cost model cannot see, so it is left on the baseline
// layout; full-tensor and f32 side outputs are not affected.
bool LayoutPropagation::proposalHasRiskyScalarSideOutput(
    const Proposal &proposal) const {
  auto hasRankOneLoopCarriedValue = [&]() {
    for (auto &it : proposal) {
      Value value = it.first;
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() != 1)
        continue;

      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
        if (forOp && blockArg.getArgNumber() > 0)
          return true;
        continue;
      }

      if (auto result = dyn_cast<OpResult>(value))
        if (isa<scf::ForOp>(result.getOwner()))
          return true;
    }
    return false;
  };

  if (!hasRankOneLoopCarriedValue())
    return false;

  // Walk forward from the proposal to find a scalar f16 writeback (a scalar
  // carries no layout, so this is purely a marker for the kernel family
  // described above, not a layout-correctness condition).
  DenseSet<Value> visited;
  SmallVector<Value> queue;
  for (auto &it : proposal)
    queue.push_back(it.first);

  while (!queue.empty()) {
    Value value = queue.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    for (Operation *user : value.getUsers()) {
      if (auto truncOp = dyn_cast<arith::TruncFOp>(user)) {
        // Match only a per-row *scalar* 16-bit side output (f16/bf16 share the
        // same register tiling); a full-tensor 16-bit output is the profitable
        // case and must not match, hence the scalar requirement.
        Type outType = truncOp.getOut().getType();
        if (!isa<RankedTensorType>(outType) &&
            (outType.isF16() || outType.isBF16())) {
          for (Operation *truncUser : truncOp.getOut().getUsers())
            if (isa<StoreOp>(truncUser))
              return true;
        }
        continue;
      }

      if (isa<StoreOp>(user))
        continue;

      if (!(user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
            user->hasTrait<OpTrait::Elementwise>() ||
            isa<ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp, ReduceOp,
                ConvertLayoutOp>(user)))
        continue;
      if (isLayoutAnchor(user) && !isa<ReduceOp, ReshapeOp>(user))
        continue;

      for (Value result : user->getResults())
        queue.push_back(result);
    }
  }

  return false;
}

bool LayoutPropagation::proposalHasRankChangingLayoutBridge(
    const Proposal &proposal) const {
  bool hasRankChangingProducer = false;
  bool hasLayoutBridgeConvert = false;

  for (auto &it : proposal) {
    Value value = it.first;
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      continue;

    if (isa<ExpandDimsOp, BroadcastOp>(defOp)) {
      auto resultType = dyn_cast<RankedTensorType>(value.getType());
      bool changesShape =
          llvm::any_of(defOp->getOperands(), [&](Value operand) {
            auto operandType = dyn_cast<RankedTensorType>(operand.getType());
            return operandType && resultType &&
                   operandType.getShape() != resultType.getShape();
          });
      hasRankChangingProducer |= changesShape;
    }

    if (auto cvtOp = dyn_cast<ConvertLayoutOp>(defOp)) {
      Attribute srcEncoding = getTensorEncoding(cvtOp.getSrc());
      Attribute resultEncoding = getTensorEncoding(cvtOp.getResult());
      hasLayoutBridgeConvert |= srcEncoding && resultEncoding &&
                                srcEncoding != resultEncoding &&
                                isa<DistributedEncodingTrait>(srcEncoding) &&
                                isa<DistributedEncodingTrait>(resultEncoding);
    }
  }

  return hasRankChangingProducer && hasLayoutBridgeConvert;
}

bool LayoutPropagation::commitProposal(const Proposal &proposal) {
  bool changed = false;
  for (auto &it : proposal)
    changed |= addSmallComponentEncoding(it.first, it.second);
  return changed;
}

static bool isBackwardPropagationBoundaryOp(Operation *op) {
  return isLayoutAnchor(op) ||
         isa<LoadOp, LocalLoadOp, DotOp, DotScaledOp, AtomicRMWOp, AtomicCASOp,
             nvidia_gpu::WarpGroupDotOp, nvidia_gpu::WarpGroupDotWaitOp,
             nvidia_gpu::TMEMLoadOp, DescriptorLoadOp, DescriptorGatherOp,
             ReduceOp, scf::ForOp, scf::WhileOp, scf::IfOp, ReshapeOp, TransOp,
             JoinOp, SplitOp, GatherOp>(op);
}

static bool isBackwardPropagationBoundaryValue(Value value) {
  Operation *defOp = value.getDefiningOp();
  return !defOp || isBackwardPropagationBoundaryOp(defOp);
}

static bool hasBoundaryTensorOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    if (isBackwardPropagationBoundaryValue(operand))
      return true;
  }
  return false;
}

static bool isRegionFreeElementwise(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() && op->getNumRegions() == 0;
}

static bool isRegionFreeSameEncoding(Operation *op) {
  return op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() &&
         op->getNumRegions() == 0;
}

// A pure op that keeps all tensor operands and results on one identical
// encoding and shape without declaring an encoding trait (out-of-tree
// pointer-materialization helpers are the common case). It transfers layouts as
// an identity function, so retagging all its tensor operands/results together
// is type-consistent. Ops with encoding traits or transfer functions are
// excluded (handled by the regular propagation rules).
static bool isEncodingUniformPureOp(Operation *op) {
  if (op->getNumRegions() != 0 || !isMemoryEffectFree(op))
    return false;
  if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<OpTrait::Elementwise>() || isa<ConvertLayoutOp>(op))
    return false;

  Attribute encoding;
  ArrayRef<int64_t> shape;
  bool hasTensorResult = false;
  auto accumulate = [&](Type type, bool isResult) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType)
      return true; // Non-tensor values do not constrain the encoding.
    if (!tensorType.getEncoding())
      return false;
    if (encoding && (encoding != tensorType.getEncoding() ||
                     shape != tensorType.getShape()))
      return false;
    encoding = tensorType.getEncoding();
    shape = tensorType.getShape();
    hasTensorResult |= isResult;
    return true;
  };
  for (Value operand : op->getOperands())
    if (!accumulate(operand.getType(), /*isResult=*/false))
      return false;
  for (Value result : op->getResults())
    if (!accumulate(result.getType(), /*isResult=*/true))
      return false;
  return hasTensorResult && encoding != nullptr;
}

static bool isBackwardPropagationForwardUser(Operation *op) {
  if (isa<StoreOp, AtomicRMWOp, AtomicCASOp>(op))
    return true;
  return isRegionFreeElementwise(op) || isRegionFreeSameEncoding(op);
}

static bool isBackwardPropagationWritebackUser(Operation *op) {
  return isa<StoreOp, AtomicRMWOp, AtomicCASOp>(op);
}

static bool isBackwardPropagationTailUser(Operation *op, Value input) {
  if (isBackwardPropagationWritebackUser(op))
    return true;
  if (op->getNumRegions() != 0 ||
      isa<ReduceOp, ScanOp, scf::ForOp, scf::WhileOp, scf::IfOp>(op))
    return false;
  if (isBackwardPropagationForwardUser(op))
    return true;

  // Some writeback tails materialize a pointer from an index tensor (e.g.
  // local/shared pointer helpers) before the final atomic. They are not
  // elementwise but preserve the index operand's layout; treat them as
  // tail-only users so they can justify a seed without a general rule.
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || op->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return resultType && resultType.getEncoding() == inputType.getEncoding();
}

static bool reachesBackwardPropagationWriteback(Value value, int depth) {
  if (depth < 0 || !hasSingleUse(value))
    return false;

  Operation *user = value.use_begin()->getOwner();
  if (isBackwardPropagationWritebackUser(user))
    return true;
  if (!isBackwardPropagationTailUser(user, value))
    return false;

  SmallVector<Value> tensorResults;
  for (Value result : user->getResults()) {
    if (isa<RankedTensorType>(result.getType()))
      tensorResults.push_back(result);
  }
  if (tensorResults.size() != 1)
    return false;
  return reachesBackwardPropagationWriteback(tensorResults.front(), depth - 1);
}

static bool isBackwardPropagationLocalDef(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (isBackwardPropagationBoundaryOp(op))
    return false;
  if (hasBoundaryTensorOperand(op))
    return false;
  return true;
}

static bool canCreateBackwardLayout(Value value) {
  if (isFrozenLayoutValue(value))
    return false;
  Operation *defOp = value.getDefiningOp();
  return defOp && isBackwardPropagationLocalDef(defOp);
}

static bool isBackwardPropagationSourceCandidate(Operation *op) {
  if (isInsideStructuredControlFlow(op))
    return false;

  if (isRegionFreeElementwise(op))
    return !hasBoundaryTensorOperand(op);

  if (isRegionFreeSameEncoding(op))
    return !hasBoundaryTensorOperand(op);

  if (auto expandOp = dyn_cast<ExpandDimsOp>(op)) {
    Value src = expandOp.getSrc();
    if (!hasSingleUse(src) || isFrozenLayoutValue(src))
      return false;
    Operation *srcDef = src.getDefiningOp();
    if (!srcDef)
      return false;
    if (isa<ConvertLayoutOp>(srcDef))
      return true;
    if (isRegionFreeElementwise(srcDef))
      return !hasBoundaryTensorOperand(srcDef);
    if (isRegionFreeSameEncoding(srcDef))
      return !hasBoundaryTensorOperand(srcDef);
    return false;
  }

  return false;
}

// A producer cone reaching a tt.reduce within a few transparent hops marks a
// reduce writeback tail. Backward propagation cannot dissolve that boundary
// (the slice parent is pinned by the reduce input), so seeding it only splits
// one convert into two. Those tails belong to Phase 2's cost-modeled
// reduce-tail proposals instead.
static bool producerConeReachesReduce(Value value, int depth) {
  if (depth <= 0)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<ReduceOp>(def))
    return true;
  if (!(isRegionFreeElementwise(def) || isRegionFreeSameEncoding(def) ||
        isa<ExpandDimsOp, BroadcastOp>(def)))
    return false;
  for (Value operand : def->getOperands())
    if (isa<RankedTensorType>(operand.getType()) &&
        producerConeReachesReduce(operand, depth - 1))
      return true;
  return false;
}

// A convert_layout is a Phase 1b candidate when it is a single-use,
// blocked-to-blocked writeback-tail convert whose source is a non-frozen local
// producer and whose result reaches a store/atomic through a short, single-use
// local chain. This covers cheap cast/index tails without changing reductions,
// scans, anchors, or control-flow-carried values.
static bool isBackwardPropagationSeedCandidate(ConvertLayoutOp cvtOp) {
  if (isInsideStructuredControlFlow(cvtOp))
    return false;

  Attribute srcEncoding = getTensorEncoding(cvtOp.getSrc());
  Attribute dstEncoding = getTensorEncoding(cvtOp.getResult());
  if (!isa_and_nonnull<BlockedEncodingAttr>(srcEncoding) ||
      !isa_and_nonnull<BlockedEncodingAttr>(dstEncoding))
    return false;
  if (isSliceEncoding(srcEncoding) || isSliceEncoding(dstEncoding))
    return false;

  Value src = cvtOp.getSrc();
  if (isFrozenLayoutValue(src))
    return false;
#ifdef __TLE__
  // Do not seed backward propagation on a TLE cluster remote-address chain (see
  // valueOnTleRemotePointerPath); retagging it corrupts the distributed access.
  if (valueOnTleRemotePointerPath(src) ||
      valueOnTleRemotePointerPath(cvtOp.getResult()))
    return false;
#endif // __TLE__
  if (!isBackwardPropagationSourceCandidate(src.getDefiningOp()))
    return false;
  if (!hasSingleUse(src) || !hasSingleUse(cvtOp.getResult()))
    return false;

  constexpr int kMaxReduceConeDepth = 5;
  if (producerConeReachesReduce(src, kMaxReduceConeDepth))
    return false;

  constexpr int kMaxBackwardWritebackTailDepth = 4;
  return reachesBackwardPropagationWriteback(cvtOp.getResult(),
                                             kMaxBackwardWritebackTailDepth);
}

SmallVector<Value> LayoutPropagation::propagateToOperands(Value value,
                                                          LayoutInfo &info) {
  SmallVector<Value> changed;

  // Block arguments are frozen for backward propagation, and we never push past
  // an anchor.
  Operation *defOp = value.getDefiningOp();
  if (!defOp || isLayoutAnchor(defOp))
    return changed;

  // Stop at multi-use values established by the forward pass: their conflict is
  // already decided by forward propagation. Values introduced by Phase 1b stay
  // on the local chain.
  if (!backwardCreatedValues.contains(value) && !hasSingleUse(value))
    return changed;

  for (Attribute encoding : info.encodings) {
    Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
    if (!srcEncoding)
      continue;
    for (OpOperand &operand : defOp->getOpOperands()) {
      Value operandVal = operand.get();
      if (!isa<RankedTensorType>(operandVal.getType()))
        continue;
      if (addBackwardEncoding(operandVal, srcEncoding))
        changed.push_back(operandVal);
    }
  }
  return changed;
}

bool LayoutPropagation::propagateLayoutBackward() {
  if (!enableBackwardPropagation)
    return false;

  SmallVector<Value> queue;
  funcOp.walk([&](ConvertLayoutOp cvtOp) {
    if (!isBackwardPropagationSeedCandidate(cvtOp))
      return;

    Attribute srcEnc = getTensorEncoding(cvtOp.getSrc());
    Attribute dstEnc = getTensorEncoding(cvtOp.getResult());
    if (!srcEnc || !dstEnc)
      return;

    Value src = cvtOp.getSrc();
    if (addBackwardEncoding(src, dstEnc))
      queue.push_back(src);

    Value result = cvtOp.getResult();
    bool resultChanged = addBackwardEncoding(result, srcEnc);
    resultChanged = addBackwardEncoding(result, dstEnc) || resultChanged;
    if (resultChanged)
      queue.push_back(result);
  });

  bool changed = !queue.empty();

  while (!queue.empty()) {
    Value currentValue = queue.back();
    queue.pop_back();
    auto it = layouts.find(currentValue);
    if (it == layouts.end())
      continue;
    LayoutInfo info = it->second;
    SmallVector<Value> changed = propagateToOperands(currentValue, info);
    queue.insert(queue.end(), changed.begin(), changed.end());
  }
  return changed;
}

static bool isSmallComponentForbiddenOp(Operation *op) {
  return isa<DotOp, DotScaledOp, AtomicRMWOp, AtomicCASOp,
             nvidia_gpu::WarpGroupDotOp, nvidia_gpu::WarpGroupDotWaitOp,
             nvidia_gpu::TMEMLoadOp, DescriptorLoadOp, DescriptorGatherOp,
             DescriptorOpInterface, ReduceOp, ScanOp>(op);
}

static bool isSmallComponentRetaggable(Value value) {
  if (!getTensorEncoding(value))
    return false;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  if (isSmallComponentForbiddenOp(defOp))
    return false;
  if (isa<LoadOp, StoreOp>(defOp) && isExpensiveLoadOrStore(defOp))
    return false;
  if (auto gatherOp = dyn_cast<GatherOp>(defOp))
    return !gatherOp.getEfficientLayout();
  return true;
}

static bool sameTensorShape(RankedTensorType lhs, RankedTensorType rhs) {
  return lhs.getRank() == rhs.getRank() && lhs.getShape() == rhs.getShape();
}

static bool isSliceEncoding(Attribute encoding) {
  return isa_and_nonnull<SliceEncodingAttr>(encoding);
}

static bool isBlockedOrBlockedSliceEncoding(Attribute encoding) {
  if (isa_and_nonnull<BlockedEncodingAttr>(encoding))
    return true;
  auto sliceEncoding = dyn_cast_or_null<SliceEncodingAttr>(encoding);
  return sliceEncoding && isa<BlockedEncodingAttr>(sliceEncoding.getParent());
}

// A slice of a tensor-core (MMA) encoding. This is the layout of a vector
// that is expanded and broadcast onto an MMA accumulator, e.g. a per-block
// quantization scale or a bias row multiplied into a dot result.
static bool isMmaSliceEncoding(Attribute encoding) {
  auto sliceEncoding = dyn_cast_or_null<SliceEncodingAttr>(encoding);
  return sliceEncoding && isa<MmaEncodingTrait>(sliceEncoding.getParent());
}

// Number of truly consecutive elements a thread owns along the fastest memory
// dimension. Uses getContigPerThread rather than getElemsPerThread, which
// counts strided repetitions too and overstates the store vector width for MMA
// layouts.
static unsigned getContigAlongMemoryOrder(RankedTensorType type) {
  SmallVector<unsigned> order = getOrderForMemory(type);
  SmallVector<unsigned> contigPerThread = getContigPerThread(type);
  if (order.empty() || contigPerThread.empty())
    return 1;
  unsigned dim = order.front();
  return dim < contigPerThread.size() ? contigPerThread[dim] : 1;
}

// A writeback retag (ptr/value/mask onto the value's layout) is profitable only
// when the layout still writes along the same fastest memory dimension with a
// reasonable per-thread width; otherwise it trades a convert for a worse global
// access. With `allowNarrowerContiguity` (tensor-core writebacks) a narrower
// per-thread contiguity is accepted while each thread still writes a full
// 32-bit word, since removing the convert's smem round trip dominates.
static bool preservesWritebackMemoryAccess(RankedTensorType ptrType,
                                           RankedTensorType valueType,
                                           Attribute targetEncoding,
                                           bool allowNarrowerContiguity) {
  if (!ptrType || !valueType || !targetEncoding ||
      !isa<DistributedEncodingTrait>(targetEncoding))
    return false;
  if (!sameTensorShape(ptrType, valueType))
    return false;

  if (ptrType.getEncoding() == targetEncoding)
    return true;

  RankedTensorType targetType = valueType.cloneWithEncoding(targetEncoding);
  SmallVector<unsigned> currentOrder = getOrderForMemory(ptrType);
  SmallVector<unsigned> targetOrder = getOrderForMemory(targetType);
  if (currentOrder.empty() || targetOrder.empty() ||
      currentOrder.front() != targetOrder.front())
    return false;

  unsigned targetContig = getContigAlongMemoryOrder(targetType);
  if (targetContig >= getContigAlongMemoryOrder(ptrType))
    return true;
  if (!allowNarrowerContiguity)
    return false;
  Type elementType = valueType.getElementType();
  unsigned elementBits =
      elementType.isIntOrFloat() ? elementType.getIntOrFloatBitWidth() : 0;
  return targetContig * elementBits >= 32;
}

static bool preservesStoreMemoryAccess(StoreOp storeOp,
                                       Attribute targetEncoding) {
  return preservesWritebackMemoryAccess(
      dyn_cast<RankedTensorType>(storeOp.getPtr().getType()),
      dyn_cast<RankedTensorType>(storeOp.getValue().getType()), targetEncoding,
      /*allowNarrowerContiguity=*/false);
}

// Number of threads that own at least one element of `type`, i.e. the store's
// effective lane parallelism. Each dimension distributes
// `threadsPerWarp[d] * warpsPerCTA[d]` threads capped at the dimension size;
// the product over dimensions is returned.
static int64_t getStoreThreadParallelism(RankedTensorType type) {
  auto layout = dyn_cast<DistributedEncodingTrait>(type.getEncoding());
  if (!layout)
    return 0;
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<unsigned> threadsPerWarp = getThreadsPerWarp(type);
  SmallVector<unsigned> warpsPerCTA = getWarpsPerCTA(type);
  if (threadsPerWarp.size() != shape.size() ||
      warpsPerCTA.size() != shape.size())
    return 0;
  int64_t parallelism = 1;
  for (unsigned d = 0, e = shape.size(); d < e; ++d) {
    int64_t threadsAlongDim =
        int64_t(threadsPerWarp[d]) * int64_t(warpsPerCTA[d]);
    parallelism *= std::min<int64_t>(shape[d], threadsAlongDim);
  }
  return parallelism;
}

// A store retag only helps when the target layout still coalesces the write.
// Two failure modes are rejected:
//   1. The fastest memory dimension collapses to a single element (e.g. a
//      reduction result written as an Nx1 tensor gains no coalescing).
//   2. A store with a degenerate (size-1) dimension whose target spreads it
//      across fewer cooperating threads than the address's own layout (a
//      per-row scalar side output). Non-degenerate stores are real
//      vectors/matrices and left to the cost model.
static bool storeRetagCoalesces(RankedTensorType ptrType,
                                Attribute targetEncoding) {
  if (!ptrType || !targetEncoding)
    return false;
  RankedTensorType targetType = ptrType.cloneWithEncoding(targetEncoding);
  SmallVector<unsigned> order = getOrderForMemory(targetType);
  if (order.empty())
    return false;
  if (ptrType.getDimSize(order.front()) <= 1)
    return false;
  // Failure mode 2 (see above): the parallelism check only guards stores with a
  // size-1 degenerate dimension; real vector/matrix stores are left to the cost
  // model.
  bool hasDegenerateDim = llvm::is_contained(ptrType.getShape(), 1);
  if (hasDegenerateDim && getStoreThreadParallelism(targetType) <
                              getStoreThreadParallelism(ptrType))
    return false;
  return true;
}

// The access-pattern checks above compare encoding orders, assuming the fastest
// dimension reflects real contiguity. When the address has no contiguity there
// anyway (strided windowed access, or a size-1 fastest dimension), the order
// protects nothing and an order-changing retag is safe. Contiguity comes from
// the axis analysis, so genuinely coalesced accesses are never relaxed.
static bool isOrderIndifferentAccess(ModuleAxisInfoAnalysis *axisInfo,
                                     Value ptr) {
  if (!axisInfo)
    return false;
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
  if (!ptrType)
    return false;
  AxisInfo *info = axisInfo->getAxisInfo(ptr);
  if (!info)
    return false;
  SmallVector<unsigned> order = getOrderForMemory(ptrType);
  if (order.empty())
    return false;
  unsigned dim = order.front();
  // A size-1 dimension is never really contiguous regardless of what the
  // analysis reports for it.
  int64_t contiguity =
      std::min<int64_t>(info->getContiguity(dim), ptrType.getDimSize(dim));
  return contiguity <= 1;
}

bool LayoutPropagation::solveSmallComponents() {
  if (!enableSmallComponentSolving)
    return false;

  constexpr int kMaxStoreTailDepth = 8;
  constexpr int kMaxProducerClosureDepth = 64;

  // A pure data-movement kernel (no reduction/scan/atomic/matmul) is dominated
  // by its global accesses, so keeping a bridging convert is cheaper than
  // amplifying a scatter store (see the Load/Store passthrough guard below).
  const bool memoryMovementOnly = !functionHasComputeAnchors(funcOp);

  bool changed = false;
  // A proposal value adjacent to a tt.join/tt.split radix step (a radix
  // butterfly boundary, whose lowering cost is set by the surrounding layout).
  auto proposalBridgesRadixSplitMerge = [](const Proposal &proposal) {
    for (auto &it : proposal) {
      Value value = it.first;
      if (Operation *defOp = value.getDefiningOp())
        if (isa<JoinOp, SplitOp>(defOp))
          return true;
      for (Operation *user : value.getUsers())
        if (isa<JoinOp, SplitOp>(user))
          return true;
    }
    return false;
  };
  // A value defined (or carried) inside a loop marks the proposal as
  // loop-resident: the convert it removes repeats every iteration.
  auto proposalHasLoopResidentValue = [](const Proposal &proposal) {
    for (auto &it : proposal) {
      Operation *scope;
      if (Operation *defOp = it.first.getDefiningOp())
        scope = defOp->getParentOp();
      else
        scope = cast<BlockArgument>(it.first).getOwner()->getParentOp();
      for (Operation *op = scope; op; op = op->getParentOp())
        if (isa<scf::ForOp, scf::WhileOp>(op))
          return true;
    }
    return false;
  };

  auto commitIfProfitable = [&](Proposal &proposal, bool requireBenefit,
                                bool rejectReachableReductionOrScan = true,
                                bool restrictWeakStoreProposals = false) {
    if (proposal.empty())
      return false;
#ifdef __TLE__
    // Skip any component on a TLE cluster remote-address chain; retagging it
    // corrupts the remote access (see valueOnTleRemotePointerPath).
    for (auto &it : proposal)
      if (valueOnTleRemotePointerPath(it.first))
        return false;
#endif // __TLE__
    int removedConverts = countConvertsRemovedByProposal(proposal);
    if (removedConverts == 0)
      return false;
    // A "weak" proposal removes a single one-shot convert outside any loop; its
    // layout ripple can cost more than it saves, so only loop-resident or
    // multi-convert components are admitted.
    bool isWeakProposal =
        removedConverts < 2 && !proposalHasLoopResidentValue(proposal);
    if (isWeakProposal &&
        (restrictWeakStoreProposals || proposalAmplifiesScatterStore(proposal)))
      return false;
    // In a pure data-movement kernel, amplifying a scatter store to remove a
    // convert is a loss when the stored value is only a copy of a coalesced
    // load (the bridging convert is an on-chip reshuffle cheaper than the
    // amplified global store). Compute kernels are excluded by the two
    // predicates.
    if (restrictWeakStoreProposals && memoryMovementOnly &&
        proposalAmplifiesScatterStore(proposal) &&
        proposalStoresOnlyLoadCopies(proposal))
      return false;
    // In a pure data-movement kernel a tt.join/tt.split radix butterfly has no
    // compute anchor to amortize a relayout; collapsing its bridging convert
    // only shifts the layout boundary (unpriced by the convert-only cost model)
    // and can turn a register-local pack/unpack into a decoalescing shuffle.
    if (memoryMovementOnly && proposalBridgesRadixSplitMerge(proposal))
      return false;
    if (removedConverts < 2 && proposalDecoalescesContiguousStore(proposal))
      return false;
    if (!proposalHasOnlyPairedForBlockArgs(proposal))
      return false;
    if (proposalHasRiskyScalarSideOutput(proposal))
      return false;
    if (rejectReachableReductionOrScan
            ? proposalTouchesReductionOrScan(proposal)
            : proposalDirectlyTouchesReductionOrScan(proposal))
      return false;
    if (!rejectReachableReductionOrScan &&
        proposalHasRankChangingLayoutBridge(proposal) &&
        proposalTouchesReductionOrScan(proposal))
      return false;
    if (requireBenefit && !shouldAcceptProposal(proposal))
      return false;
    return commitProposal(proposal);
  };

  auto collectRetaggableStoreValue = [&](Proposal &proposal, Value value,
                                         Attribute encoding) {
    if (!isa<RankedTensorType>(value.getType()))
      return true;
    if (!getTensorEncoding(value))
      return false;
    if (!isSmallComponentRetaggable(value))
      return false;
    return addProposalValue(proposal, value, encoding);
  };

  // A pointer/mask operand may be retagged only when its whole producer chain
  // is retaggable too; otherwise the retag strands a convert on an anchored
  // chain (e.g. an address from a load) that backward remat cannot remove. The
  // chain is only probed, not committed, to keep shared producers out.
  auto collectStoreAddressOperand = [&](Proposal &proposal, Value value,
                                        Attribute encoding) {
    if (!isa<RankedTensorType>(value.getType()))
      return true;
    // A loop-carried operand must commit its paired loop values together, so
    // its closure is collected directly into the proposal.
    if (isa<BlockArgument>(value))
      return collectProducerClosure(proposal, value, encoding,
                                    kMaxProducerClosureDepth,
                                    /*stopAtConvertSrc=*/true);
    Proposal chainProbe;
    if (!collectProducerClosure(chainProbe, value, encoding,
                                kMaxProducerClosureDepth,
                                /*stopAtConvertSrc=*/true))
      return false;
    if (!isSmallComponentRetaggable(value))
      return false;
    return addProposalValue(proposal, value, encoding);
  };

  // The terminal of a store tail: a tt.store or a tt.atomic_rmw with unused
  // result. All tensor operands (and an atomic's unused result) are retagged
  // together to keep the operand encodings consistent; the incoming chain joins
  // via the direct retaggable check, the rest via the producer-closure probe.
  auto collectWritebackTerminal = [&](Proposal &proposal, Operation *user,
                                      Value incoming,
                                      Attribute encoding) -> bool {
    if (!isBlockedOrBlockedSliceEncoding(encoding))
      return false;
    auto collectOperand = [&](Value operand) {
      if (operand == incoming)
        return collectRetaggableStoreValue(proposal, operand, encoding);
      return collectStoreAddressOperand(proposal, operand, encoding);
    };
    if (auto storeOp = dyn_cast<StoreOp>(user)) {
      if (!preservesStoreMemoryAccess(storeOp, encoding))
        return false;
      if (!storeRetagCoalesces(
              dyn_cast<RankedTensorType>(storeOp.getPtr().getType()), encoding))
        return false;
      if (!collectOperand(storeOp.getValue()) ||
          !collectOperand(storeOp.getPtr()))
        return false;
      if (storeOp.getMask())
        return collectOperand(storeOp.getMask());
      return true;
    }
    auto atomicOp = dyn_cast<AtomicRMWOp>(user);
    if (!atomicOp || !atomicOp.getResult().use_empty())
      return false;
    auto ptrType = dyn_cast<RankedTensorType>(atomicOp.getPtr().getType());
    auto valType = dyn_cast<RankedTensorType>(atomicOp.getVal().getType());
    if (!ptrType || !valType)
      return false;
    if (!preservesWritebackMemoryAccess(ptrType, valType, encoding,
                                        /*allowNarrowerContiguity=*/false))
      return false;
    if (!collectOperand(atomicOp.getVal()) ||
        !collectOperand(atomicOp.getPtr()))
      return false;
    if (Value mask = atomicOp.getMask())
      if (!collectOperand(mask))
        return false;
    return addProposalValue(proposal, atomicOp.getResult(), encoding);
  };

  auto collectDownstreamStoreTail = [&](Proposal &proposal, Value start,
                                        Attribute startEncoding) {
    if (!isBlockedOrBlockedSliceEncoding(startEncoding))
      return false;
    Value value = start;
    Attribute encoding = startEncoding;
    for (int depth = 0; depth < kMaxStoreTailDepth; ++depth) {
      if (!encoding || !hasSingleUse(value))
        return false;
      Operation *user = *value.getUsers().begin();

      if (auto cvtOp = dyn_cast<ConvertLayoutOp>(user)) {
        value = cvtOp.getResult();
        if (!addProposalValue(proposal, value, encoding))
          return false;
        continue;
      }
      if (auto expandOp = dyn_cast<ExpandDimsOp>(user)) {
        Attribute dstEncoding = inferDstEncoding(expandOp, encoding);
        if (!dstEncoding)
          return false;
        value = expandOp.getResult();
        encoding = dstEncoding;
        if (!addProposalValue(proposal, value, encoding))
          return false;
        continue;
      }
      if (auto broadcastOp = dyn_cast<BroadcastOp>(user)) {
        value = broadcastOp.getResult();
        if (!hasSingleUse(value))
          return false;
        if (!addProposalValue(proposal, value, encoding))
          return false;
        continue;
      }
      if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
              arith::TruncFOp, BitcastOp>(user)) {
        if (user->getNumResults() != 1 || !hasSingleUse(user->getResult(0)))
          return false;
        value = user->getResult(0);
        if (!addProposalValue(proposal, value, encoding))
          return false;
        continue;
      }
      if (isEncodingUniformPureOp(user)) {
        if (user->getNumResults() != 1 || !hasSingleUse(user->getResult(0)))
          return false;
        value = user->getResult(0);
        if (!addProposalValue(proposal, value, encoding))
          return false;
        continue;
      }
      // Expensive (coalesced) writebacks are fine to retag as well: the
      // terminal collector only accepts encodings that preserve the memory
      // access pattern.
      if (isa<StoreOp, AtomicRMWOp>(user))
        return collectWritebackTerminal(proposal, user, value, encoding);
      return false;
    }
    return false;
  };

  auto collectConvertStoreTail = [&](Proposal &proposal,
                                     ConvertLayoutOp cvtOp) {
    Value result = cvtOp.getResult();
    if (!hasSingleUse(result))
      return false;
    Operation *user = *result.getUsers().begin();
    if (!isa<StoreOp, AtomicRMWOp>(user))
      return false;

    Value src = cvtOp.getSrc();
    if (!getTensorEncoding(src))
      return false;
    Attribute encoding = getTensorEncoding(src);
    if (!addProposalValue(proposal, result, encoding))
      return false;
    return collectWritebackTerminal(proposal, user, result, encoding);
  };

  auto collectProducerSideConvert = [&](Proposal &proposal,
                                        ConvertLayoutOp cvtOp) {
    Value src = cvtOp.getSrc();
    Value result = cvtOp.getResult();
    // The source chain must be on blocked layouts; the target may also be an
    // MMA slice (the accumulator-scale/bias pattern: a loaded vector expanded
    // and broadcast onto a dot result). Retagging the producer chain removes a
    // convert that would round-trip through smem next to the dot. Loads stay
    // guarded by the closure's access-preservation check.
    if (!isBlockedOrBlockedSliceEncoding(getTensorEncoding(src)))
      return false;
    Attribute targetEncoding = getTensorEncoding(result);
    if (!isBlockedOrBlockedSliceEncoding(targetEncoding) &&
        !isMmaSliceEncoding(targetEncoding))
      return false;
    if (!hasSingleUse(src))
      return false;
    // An MMA-derived slice may replicate the value across warps with many
    // elements per thread. Committing a loop-resident chain onto it trades the
    // convert's smem broadcast for a far larger per-iteration register/load
    // footprint; bound it like the duplicable-load path does.
    if (isMmaSliceEncoding(targetEncoding) &&
        (cvtOp->getParentOfType<scf::ForOp>() ||
         cvtOp->getParentOfType<scf::WhileOp>())) {
      auto resultType = cast<RankedTensorType>(result.getType());
      Type elementType = resultType.getElementType();
      int64_t elementBits =
          elementType.isIntOrFloat() ? elementType.getIntOrFloatBitWidth() : 64;
      if (int64_t(getTotalElemsPerThread(
              resultType.cloneWithEncoding(targetEncoding))) *
              elementBits >
          256)
        return false;
    }
    // Only solve for final layout boundaries: if the result flows into another
    // convert through layout-transparent ops, the target encoding is a
    // transient stop and committing the chain to it merely relocates the
    // convert.
    if (convertResultReachesAnotherConvert(cvtOp))
      return false;

    // The producer chain may live inside a loop (the closure pairs
    // loop-carried values with their results) and may end at loads whose
    // access pattern is preserved; collectProducerClosure arbitrates both.
    Operation *srcDef = src.getDefiningOp();
    if (!srcDef || isSmallComponentForbiddenOp(srcDef) ||
        isa<scf::ForOp, scf::WhileOp, scf::IfOp>(srcDef))
      return false;

    if (!collectProducerClosure(proposal, src, targetEncoding,
                                kMaxProducerClosureDepth,
                                /*stopAtConvertSrc=*/true))
      return false;
    return addProposalValue(proposal, result, targetEncoding);
  };

  auto proposalsOverlapAndCompatible = [](const Proposal &lhs,
                                          const Proposal &rhs) -> bool {
    bool overlaps = false;
    for (auto &it : lhs) {
      auto rhsIt = rhs.find(it.first);
      if (rhsIt == rhs.end())
        continue;
      overlaps = true;
      if (rhsIt->second != it.second)
        return false;
    }
    return overlaps;
  };

  auto mergeProposalInto = [](Proposal &dst, const Proposal &src) {
    for (auto &it : src) {
      auto [dstIt, inserted] = dst.insert({it.first, it.second});
      (void)inserted;
      assert(dstIt->second == it.second &&
             "only compatible proposals should be merged");
    }
  };

  auto commitMergedProposals = [&](SmallVector<Proposal> proposals,
                                   bool rejectReachableReductionOrScan = true,
                                   bool restrictWeakStoreProposals = false) {
    bool localChanged = false;
    bool merged = true;
    while (merged) {
      merged = false;
      for (size_t i = 0; i < proposals.size() && !merged; ++i) {
        for (size_t j = i + 1; j < proposals.size(); ++j) {
          if (!proposalsOverlapAndCompatible(proposals[i], proposals[j]))
            continue;
          mergeProposalInto(proposals[i], proposals[j]);
          proposals.erase(proposals.begin() + j);
          merged = true;
          break;
        }
      }
    }
    for (Proposal &proposal : proposals)
      localChanged |= commitIfProfitable(proposal, /*requireBenefit=*/true,
                                         rejectReachableReductionOrScan,
                                         restrictWeakStoreProposals);
    return localChanged;
  };

  // The patterns below are processed as ordered stages. Each stage collects
  // candidate proposals, merges the overlapping/compatible ones, and commits
  // those that the cost model accepts. Committing records a preferred encoding
  // that later stages observe, so the stage order is significant and the stages
  // are not interchangeable (do not fold them into one walk).

  auto collectResultStoreTailProposals = [&](auto op,
                                             SmallVectorImpl<Proposal> &out) {
    for (Value result : op->getResults()) {
      if (!getTensorEncoding(result) || !hasSingleUse(result))
        continue;
      Proposal proposal;
      if (collectDownstreamStoreTail(proposal, result,
                                     getTensorEncoding(result)))
        out.push_back(std::move(proposal));
    }
  };

  // The three writeback-tail stages restrict weak proposals (a single one-shot
  // convert outside loops): their upside is bounded to one convert per CTA,
  // whereas loop-resident or multi-convert components are the profitable case.

  // Reduce/slice writeback.
  SmallVector<Proposal> reduceTailProposals;
  funcOp.walk([&](ReduceOp reduceOp) {
    if (reduceOp->getNumResults() != 1)
      return;
    collectResultStoreTailProposals(reduceOp, reduceTailProposals);
  });
  changed |= commitMergedProposals(std::move(reduceTailProposals),
                                   /*rejectReachableReductionOrScan=*/true,
                                   /*restrictWeakStoreProposals=*/true);

  // Loop result writeback tails.
  SmallVector<Proposal> loopTailProposals;
  funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->getNumResults() != 1)
      return;
    collectResultStoreTailProposals(forOp, loopTailProposals);
  });
  changed |= commitMergedProposals(std::move(loopTailProposals),
                                   /*rejectReachableReductionOrScan=*/true,
                                   /*restrictWeakStoreProposals=*/true);

  // Direct cheap store tails.
  SmallVector<Proposal> storeTailProposals;
  funcOp.walk([&](ConvertLayoutOp cvtOp) {
    Value src = cvtOp.getSrc();
    Value result = cvtOp.getResult();
    if (!getTensorEncoding(src) || !getTensorEncoding(result) ||
        !hasSingleUse(result))
      return;
    Proposal proposal;
    if ((hasSingleUse(src) &&
         collectDownstreamStoreTail(proposal, src, getTensorEncoding(src))) ||
        collectConvertStoreTail(proposal, cvtOp))
      storeTailProposals.push_back(std::move(proposal));
  });
  changed |= commitMergedProposals(std::move(storeTailProposals),
                                   /*rejectReachableReductionOrScan=*/true,
                                   /*restrictWeakStoreProposals=*/true);

  // Producer-side local chains ending at a convert_layout.
  SmallVector<Proposal> producerSideProposals;
  funcOp.walk([&](ConvertLayoutOp cvtOp) {
    Proposal proposal;
    if (collectProducerSideConvert(proposal, cvtOp))
      producerSideProposals.push_back(std::move(proposal));
  });
  changed |= commitMergedProposals(std::move(producerSideProposals),
                                   /*rejectReachableReductionOrScan=*/false);

  return changed;
}

// Enumerate the (value, encoding) pairs that propagating `enc` across `use`
// would assign, mirroring propagateToUsers' dispatch (scf carried values,
// WarpGroupDotWait, gather indices, and encoding-transferring ops). The cost
// model uses this to count the converts a candidate encoding implies.
static void
collectUseTargets(OpOperand &use, Attribute enc,
                  SmallVectorImpl<std::pair<Value, Attribute>> &targets) {
  Operation *user = use.getOwner();

  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    targets.push_back({forOp.getTiedLoopRegionIterArg(&use), enc});
    targets.push_back({forOp.getTiedLoopResult(&use), enc});
    return;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
    targets.push_back(
        {whileOp.getBeforeArguments()[use.getOperandNumber()], enc});
    return;
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    auto parent = yieldOp->getParentOp();
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
      targets.push_back({parent->getResult(use.getOperandNumber()), enc});
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      targets.push_back({forOp.getRegionIterArg(use.getOperandNumber()), enc});
    if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
      targets.push_back(
          {whileOp.getBeforeArguments()[use.getOperandNumber()], enc});
    return;
  }
  if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
    auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
    unsigned argIndex = use.getOperandNumber() - 1;
    targets.push_back({whileOp.getAfterArguments()[argIndex], enc});
    targets.push_back({whileOp->getResult(argIndex), enc});
    return;
  }
  if (isa<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
    Value result = user->getResult(use.getOperandNumber());
    if (getTensorEncoding(result))
      targets.push_back({result, enc});
    return;
  }
  if (auto gatherOp = dyn_cast<GatherOp>(user)) {
    if (!gatherOp.getEfficientLayout() &&
        &use == &gatherOp.getIndicesMutable()) {
      Attribute dstEnc = inferDstEncoding(user, enc);
      if (dstEnc)
        targets.push_back({gatherOp.getResult(), dstEnc});
      return;
    }
  }
  if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      user->hasTrait<OpTrait::Elementwise>() ||
      isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
          ConvertLayoutOp>(user)) {
    for (Value result : user->getResults()) {
      if (!getTensorEncoding(result))
        continue;
      Attribute dstEnc =
          isa<ConvertLayoutOp>(user) ? enc : inferDstEncoding(user, enc);
      if (dstEnc)
        targets.push_back({result, dstEnc});
    }
  }
}

bool LayoutPropagation::isCompatibleWithEncoding(Value value,
                                                 Attribute encoding) const {
  if (auto it = layouts.find(value); it != layouts.end())
    return it->second.encodings.contains(encoding);

  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return !tensorType || tensorType.getEncoding() == encoding;
}

// Cost of materializing `value` in `encoding`: the smem-round-trip byte
// estimate (clamped to >= 1), or 0 when `value` already carries that encoding.
int64_t LayoutPropagation::getValueConversionCost(Value value,
                                                  Attribute encoding) const {
  if (!encoding)
    return 0;
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || tensorType.getEncoding() == encoding)
    return 0;
  int64_t cost = 32 * getByteCount(value, 32, 32);
  return cost == 0 ? 1 : cost;
}

// Cost charged on the producer side if `value` is chosen in `encoding`:
// re-deriving its defining op in that layout converts the operands that do not
// already resolve to the inferred source encoding. Expensive loads and ops with
// no usable transfer function contribute nothing (their layout is frozen).
int64_t
LayoutPropagation::getDefiningOpConversionCost(Value value,
                                               Attribute encoding) const {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return 0;

  if (isa<ConvertLayoutOp>(defOp)) {
    Value src = defOp->getOperand(0);
    return isCompatibleWithEncoding(src, encoding)
               ? 0
               : getValueConversionCost(src, encoding);
  }
  if (isa<LoadOp>(defOp) && isExpensiveLoadOrStore(defOp))
    return 0;

  if (!(defOp->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        defOp->hasTrait<OpTrait::Elementwise>() ||
        isa<LoadOp, ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            GatherOp, nvidia_gpu::WarpGroupDotWaitOp>(defOp)))
    return 0;

  Attribute srcEncoding = inferSrcEncoding(defOp, encoding);
  if (!srcEncoding)
    return 0;

  int64_t cost = 0;
  for (Value operand : defOp->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    if (!isCompatibleWithEncoding(operand, srcEncoding))
      cost += getValueConversionCost(operand, srcEncoding);
  }
  return cost;
}

int64_t LayoutPropagation::getUseConversionCost(Value value, OpOperand &use,
                                                Attribute encoding,
                                                int64_t cvtUnitCost) const {
  Operation *user = use.getOwner();

  if (backwardCreatedValues.contains(value)) {
    if (auto cvtOp = dyn_cast<ConvertLayoutOp>(user)) {
      Attribute cvtDstEnc = getTensorEncoding(cvtOp.getResult());
      if (!cvtDstEnc)
        return 0;
      return encoding == cvtDstEnc ? 0 : cvtUnitCost;
    }
  }

  SmallVector<std::pair<Value, Attribute>> targets;
  collectUseTargets(use, encoding, targets);

  int64_t cost = 0;
  for (auto &[targetVal, dstEnc] : targets)
    if (!isCompatibleWithEncoding(targetVal, dstEnc))
      cost += cvtUnitCost;

  // Ops without tensor results, such as tt.store, still keep their operand
  // types fixed during rewrite.  A candidate that differs from the current use
  // type will need a convert on this use.
  if (targets.empty() &&
      (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
       user->hasTrait<OpTrait::SameLoadStoreOperandsEncoding>())) {
    auto useType = dyn_cast<RankedTensorType>(value.getType());
    if (useType && useType.getEncoding() != encoding)
      cost += cvtUnitCost;
  }

  return cost;
}

// Cost-based resolution only runs when every candidate is a blocked encoding;
// conflicts mixing MMA/slice/dot-operand encodings keep the legacy preference
// rule, whose hardware-driven choices the byte-count model cannot weigh.
bool LayoutPropagation::shouldUseCostBasedResolution(
    const LayoutInfo &info) const {
  return llvm::all_of(info.encodings, [](Attribute enc) {
    return isa<BlockedEncodingAttr>(enc);
  });
}

bool LayoutPropagation::isExtensionTouchedValue(Value value) const {
  return backwardTouchedValues.contains(value) ||
         smallComponentPreferredEncoding.contains(value);
}
#endif // __FLAGTREE_RLC_ENHANCE__

void LayoutPropagation::resolveConflicts() {
  for (auto &it : layouts) {
#ifdef __FLAGTREE_RLC_ENHANCE__
    Value value = it.first;
    Operation *defOp = value.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    if (auto preferred = smallComponentPreferredEncoding.lookup(value)) {
      if (info.encodings.contains(preferred)) {
        info.encodings.clear();
        info.encodings.insert(preferred);
        continue;
      }
    }
    if (!enableCostBasedResolution || !isExtensionTouchedValue(value) ||
        !shouldUseCostBasedResolution(info)) {
      Attribute encoding = *info.encodings.begin();
      bool isLoadOrStore =
          defOp && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(defOp);
      for (Attribute e : info.encodings) {
        if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
            (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
          encoding = e;
          break;
        }
      }
      info.encodings.clear();
      info.encodings.insert(encoding);
      continue;
    }

    int64_t cvtUnitCost = 32 * getByteCount(value, 32, 32);
    if (cvtUnitCost == 0)
      cvtUnitCost = 1;

    SmallVector<std::pair<Attribute, int64_t>> candidateCosts;
    for (Attribute enc : info.encodings) {
      int64_t cost = getDefiningOpConversionCost(value, enc);

      for (OpOperand &use : value.getUses())
        cost += getUseConversionCost(value, use, enc, cvtUnitCost);

      candidateCosts.push_back({enc, cost});
    }

    int64_t minCost = candidateCosts[0].second;
    for (auto &[enc, cost] : candidateCosts)
      minCost = std::min(minCost, cost);

    SmallVector<Attribute> tied;
    for (auto &[enc, cost] : candidateCosts)
      if (cost == minCost)
        tied.push_back(enc);

    // On a cost tie there is no concrete benefit in changing the layout, so
    // prefer the value's current encoding to keep the rewritten IR identical
    // to the baseline. Only fall back to the legacy preference when the
    // current encoding is not among the tied candidates.
    Attribute bestEncoding = tied[0];
    if (tied.size() > 1) {
      if (Attribute currentEncoding = getTensorEncoding(value);
          currentEncoding && llvm::is_contained(tied, currentEncoding)) {
        bestEncoding = currentEncoding;
      } else {
        bool isLoadOrStore =
            defOp && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(defOp);
        for (Attribute e : tied) {
          if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
              (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
            bestEncoding = e;
            break;
          }
        }
      }
    }
    info.encodings.clear();
    info.encodings.insert(bestEncoding);
#else  // __FLAGTREE_RLC_ENHANCE__
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    Attribute encoding = *info.encodings.begin();
    bool isLoadOrStore =
        op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
    for (Attribute e : info.encodings) {
      if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
          (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
        encoding = e;
        break;
      }
    }
    info.encodings.clear();
    info.encodings.insert(encoding);
#endif // __FLAGTREE_RLC_ENHANCE__
  }
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() { rewriteRegion(funcOp->getRegion(0)); }

bool reduceToScalar(Operation *op) {
  // For reductions returning a scalar we can change the src encoding without
  // affecting the output.
  return isa<ReduceOp>(op) && !isa<RankedTensorType>(op->getResultTypes()[0]);
}

void LayoutPropagation::rewriteRegion(Region &region) {
  std::deque<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.front();
    queue.pop_front();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = cast<RankedTensorType>(result.getType()).getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        Operation *newOp = rewriteOp(&op);
        for (Region &R : newOp->getRegions())
          queue.push_back(&R);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        rewriteYieldOp(yieldOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        rewriteConditionOp(conditionOp);
      } else if (reduceToScalar(&op)) {
        rewriteReduceToScalar(&op);
      } else if (auto assertOp = dyn_cast<AssertOp>(&op)) {
        rewriteAssertOp(assertOp);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
#ifdef __FLAGTREE_RLC_ENHANCE__
        //
        // Phase 2 may have retagged a store's value. A store needs ptr/value/
        // mask on one encoding, so move the whole store onto that layout only
        // when every tensor operand already resolves to it; otherwise keep the
        // baseline per-operand remap (which converts the value back).
        Attribute storeEncoding;
        if (enableSmallComponentSolving) {
          if (auto storeOp = dyn_cast<StoreOp>(&op)) {
            if (smallComponentPreferredEncoding.contains(storeOp.getValue())) {
              if (auto it = layouts.find(storeOp.getValue());
                  it != layouts.end()) {
                Attribute candidate = *it->second.encodings.begin();
                bool allOperandsMatch =
                    llvm::all_of(op.getOpOperands(), [&](OpOperand &operand) {
                      if (!isa<RankedTensorType>(operand.get().getType()))
                        return true;
                      auto operandIt = layouts.find(operand.get());
                      return operandIt != layouts.end() &&
                             *operandIt->second.encodings.begin() == candidate;
                    });
                if (allOperandsMatch)
                  storeEncoding = candidate;
              }
            }
          }
        }
#endif // __FLAGTREE_RLC_ENHANCE__
        for (OpOperand &operand : op.getOpOperands()) {
#ifdef __FLAGTREE_RLC_ENHANCE__
          if (storeEncoding && isa<RankedTensorType>(operand.get().getType())) {
            Value newOperand = getValueAs(operand.get(), storeEncoding);
            op.setOperand(operand.getOperandNumber(), newOperand);
            continue;
          }
#endif // __FLAGTREE_RLC_ENHANCE__
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding =
              cast<RankedTensorType>(operand.get().getType()).getEncoding();
          Value newOperand = getValueAs(operand.get(), encoding);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
#ifdef __FLAGTREE_RLC_ENHANCE__
  // A phase extension can retag a value so that an op queued for deletion still
  // has a live use; skip erasing those. With all extensions off the queue only
  // holds fully-replaced ops, so this matches the original unconditional erase.
  for (Operation *op : llvm::reverse(opToDelete)) {
    if (hasLayoutPropagationExtensions() && !allResultsHaveNoUses(op))
      continue;
    op->erase();
  }
#else  // __FLAGTREE_RLC_ENHANCE__
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
#endif // __FLAGTREE_RLC_ENHANCE__
}

void LayoutPropagation::map(Value old, Value newV) {
  rewriteMapping[{old, cast<RankedTensorType>(newV.getType()).getEncoding()}] =
      newV;
}

Value LayoutPropagation::getRewrittenValue(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return value;
  auto layoutIt = layouts.find(value);
  if (layoutIt == layouts.end()) {
    return value;
  }
  assert(layoutIt->second.encodings.size() == 1 &&
         "we should have resolved to a single encoding");
  Attribute encodingPicked = *(layoutIt->second.encodings.begin());
  if (encodingPicked == tensorType.getEncoding())
    return value;
  return rewriteMapping.at({value, encodingPicked});
}

Value LayoutPropagation::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    Value rewrittenValue = getRewrittenValue(value);
    if (cast<RankedTensorType>(rewrittenValue.getType()).getEncoding() ==
        encoding)
      return rewrittenValue;
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(rewrittenValue);
    auto tmpType = tensorType.cloneWithEncoding(encoding);
    Value converted = ConvertLayoutOp::create(rewriter, value.getLoc(), tmpType,
                                              rewrittenValue);
    // TODO: we could cache the conversion.
    return converted;
  }
  return value;
}

Operation *LayoutPropagation::cloneElementwise(OpBuilder &rewriter,
                                               Operation *op,
                                               Attribute encoding) {
  Operation *newOp = rewriter.clone(*op);

  Attribute operandEnc;
  if (op->getNumOperands() > 0) {
#ifdef __FLAGTREE_RLC_ENHANCE__
    // Encoding-uniform pure ops transfer the layout as an identity function;
    // they have no transfer function for inferSrcEncoding to consult. Only the
    // phase extensions retag such ops, so the check is gated to keep the
    // all-disabled path byte-identical to the original pass.
    if (hasLayoutPropagationExtensions() && isEncodingUniformPureOp(op))
      operandEnc = encoding;
#endif // __FLAGTREE_RLC_ENHANCE__
    for (auto operand : op->getOperands()) {
#ifdef __FLAGTREE_RLC_ENHANCE__
      if (operandEnc)
        break;
#endif // __FLAGTREE_RLC_ENHANCE__
      auto ty =
          dyn_cast<RankedTensorType>(getRewrittenValue(operand).getType());
      if (!ty)
        continue;
      auto enc = ty.getEncoding();
      if (inferDstEncoding(op, enc) == encoding) {
        operandEnc = enc;
        break;
      }
    }
    if (!operandEnc)
      operandEnc = inferSrcEncoding(op, encoding);
    assert(operandEnc);
  }

  for (OpOperand &operand : op->getOpOperands()) {
    newOp->setOperand(operand.getOperandNumber(),
                      getValueAs(operand.get(), operandEnc));
  }

  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
    auto origType = dyn_cast<RankedTensorType>(op->getResult(i).getType());
    if (!origType)
      continue;
    auto newType = origType.cloneWithEncoding(encoding);
    newOp->getResult(i).setType(newType);
  }
  return newOp;
}

Operation *LayoutPropagation::rewriteForOp(scf::ForOp forOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(forOp);
  for (auto [operand, result] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults())) {
    Value convertedOperand = operand;
    if (layouts.count(result))
      convertedOperand =
          getValueAs(operand, *layouts[result].encodings.begin());
    operands.push_back(convertedOperand);
  }
  auto newForOp =
      scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), operands);
  newForOp->setAttrs(forOp->getAttrs());
  newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->getOperations().begin(),
      forOp.getBody()->getOperations());

  for (auto [oldResult, newResult] :
       llvm::zip(forOp.getResults(), newForOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }

  for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                         newForOp.getBody()->getArguments())) {
    if (oldArg.getType() == newArg.getType()) {
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }
    map(oldArg, newArg);
  }
  return newForOp.getOperation();
}

Operation *LayoutPropagation::rewriteWhileOp(scf::WhileOp whileOp) {
  SmallVector<Value> operands;
  SmallVector<Type> returnTypes;
  OpBuilder rewriter(whileOp);
  for (auto [operand, arg] :
       llvm::zip(whileOp->getOperands(), whileOp.getBeforeArguments())) {
    Value convertedOperand = operand;
    if (layouts.count(arg))
      convertedOperand = getValueAs(operand, *layouts[arg].encodings.begin());
    operands.push_back(convertedOperand);
  }
  for (Value ret : whileOp.getResults()) {
    auto it = layouts.find(ret);
    if (it == layouts.end()) {
      returnTypes.push_back(ret.getType());
      continue;
    }
    auto origType = dyn_cast<RankedTensorType>(ret.getType());
    auto newType = origType.cloneWithEncoding(it->second.encodings[0]);
    returnTypes.push_back(newType);
  }

  auto newWhileOp =
      scf::WhileOp::create(rewriter, whileOp.getLoc(), returnTypes, operands);
  SmallVector<Type> argsTypesBefore;
  for (Value operand : operands)
    argsTypesBefore.push_back(operand.getType());
  SmallVector<Location> bbArgLocsBefore(argsTypesBefore.size(),
                                        whileOp.getLoc());
  SmallVector<Location> bbArgLocsAfter(returnTypes.size(), whileOp.getLoc());
  rewriter.createBlock(&newWhileOp.getBefore(), {}, argsTypesBefore,
                       bbArgLocsBefore);
  rewriter.createBlock(&newWhileOp.getAfter(), {}, returnTypes, bbArgLocsAfter);

  for (int i = 0; i < whileOp.getNumRegions(); ++i) {
    newWhileOp->getRegion(i).front().getOperations().splice(
        newWhileOp->getRegion(i).front().getOperations().begin(),
        whileOp->getRegion(i).front().getOperations());
  }

  auto remapArg = [&](Value oldVal, Value newVal) {
    if (oldVal.getType() == newVal.getType())
      oldVal.replaceAllUsesWith(newVal);
    else
      map(oldVal, newVal);
  };
  for (auto [oldResult, newResult] :
       llvm::zip(whileOp.getResults(), newWhileOp.getResults()))
    remapArg(oldResult, newResult);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getBeforeArguments(), newWhileOp.getBeforeArguments()))
    remapArg(oldArg, newArg);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getAfterArguments(), newWhileOp.getAfterArguments()))
    remapArg(oldArg, newArg);
  return newWhileOp.getOperation();
}

Operation *LayoutPropagation::rewriteIfOp(scf::IfOp ifOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(ifOp);
  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0, e = ifOp->getNumResults(); i < e; ++i) {
    auto it = layouts.find(ifOp->getResult(i));
    if (it == layouts.end())
      continue;
    auto origType = cast<RankedTensorType>(ifOp->getResult(i).getType());
    Attribute encoding = *(it->second.encodings.begin());
    newResultTypes[i] = origType.cloneWithEncoding(encoding);
  }
  auto newIfOp = scf::IfOp::create(rewriter, ifOp.getLoc(), newResultTypes,
                                   ifOp.getCondition(), true, true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  for (auto [oldResult, newResult] :
       llvm::zip(ifOp.getResults(), newIfOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }
  return newIfOp.getOperation();
}

void LayoutPropagation::rewriteYieldOp(scf::YieldOp yieldOp) {
  Operation *parentOp = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    Type yieldType = operand.get().getType();
    if (isa<scf::ForOp, scf::IfOp>(parentOp))
      yieldType = parentOp->getResult(operand.getOperandNumber()).getType();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      yieldType =
          whileOp.getBeforeArguments()[operand.getOperandNumber()].getType();
    auto tensorType = dyn_cast<RankedTensorType>(yieldType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    yieldOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteConditionOp(scf::ConditionOp conditionOp) {
  scf::WhileOp whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  for (unsigned i = 1; i < conditionOp->getNumOperands(); ++i) {
    OpOperand &operand = conditionOp->getOpOperand(i);
    Type argType = whileOp->getResult(operand.getOperandNumber() - 1).getType();
    auto tensorType = dyn_cast<RankedTensorType>(argType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    conditionOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteReduceToScalar(Operation *reduceOp) {
  OpBuilder rewriter(reduceOp);
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second.encodings[0];
      break;
    }
  }
  if (!srcEncoding)
    return;
  for (OpOperand &operand : reduceOp->getOpOperands()) {
    Value newOperand = getValueAs(operand.get(), srcEncoding);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteAssertOp(AssertOp assertOp) {
  Attribute srcEncoding;
  // Only need to deal with the first operand which is the condition tensor.
  Value operand = assertOp->getOperand(0);
  auto it = layouts.find(operand);
  if (it == layouts.end())
    return;
  srcEncoding = it->second.encodings[0];
  Value newOperand = getValueAs(operand, srcEncoding);
  assertOp->setOperand(0, newOperand);
}

Operation *LayoutPropagation::rewriteOp(Operation *op) {
  opToDelete.insert(op);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp);
  OpBuilder rewriter(op);
#ifdef __FLAGTREE_RLC_ENHANCE__
  Attribute encoding = [&]() {
    if (!hasLayoutPropagationExtensions())
      return *layouts[op->getResult(0)].encodings.begin();

    // Pick the resolved encoding from the first tracked result: result 0 may
    // not be tracked, since the phases can retag only some results of a
    // multi-result op (e.g. a value/index reduce).
    for (Value result : op->getResults()) {
      auto it = layouts.find(result);
      if (it != layouts.end() && !it->second.encodings.empty())
        return *it->second.encodings.begin();
    }
    return Attribute();
  }();
  assert(encoding && "rewriteOp called on op with no tracked result encoding");
#else  // __FLAGTREE_RLC_ENHANCE__
  Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
#endif // __FLAGTREE_RLC_ENHANCE__
  if (auto convertOp = dyn_cast<ConvertLayoutOp>(op)) {
    Attribute srcEncoding = convertOp.getSrc().getType().getEncoding();
    auto it = layouts.find(convertOp.getSrc());
    if (it != layouts.end())
      srcEncoding = *(it->second.encodings.begin());
    Value src = getValueAs(convertOp.getSrc(), srcEncoding);
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
#ifdef __FLAGTREE_RLC_ENHANCE__
    // When a phase retagged the source onto the result encoding this becomes
    // an identity convert; keep emitting it here (at the original position)
    // and let the canonicalizer fold it, so the surrounding IR stays
    // byte-identical to the baseline whenever no convert is actually removed.
#endif // __FLAGTREE_RLC_ENHANCE__
    auto newType = tensorType.cloneWithEncoding(encoding);
    auto cvt = ConvertLayoutOp::create(rewriter, op->getLoc(), newType, src);
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (canFoldIntoConversion(op, encoding)) {
    Operation *newOp = rewriter.clone(*op);
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    auto cvt = ConvertLayoutOp::create(rewriter, op->getLoc(), newType,
                                       newOp->getResult(0));
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
#ifdef __FLAGTREE_RLC_ENHANCE__
  // The phase extensions can assign a non-anchor (cheap) load a new result
  // encoding, e.g. backward propagation pulling a store's layout up to the
  // load. Rewrite it in place by retagging its tensor operands; the elementwise
  // path below has no transfer function for a load and would hit the fatal
  // error.
  if (enableCostBasedResolution) {
    if (auto loadOp = dyn_cast<LoadOp>(op)) {
      Operation *newOp = rewriter.clone(*op);
      for (OpOperand &operand : op->getOpOperands()) {
        Value oldOperand = operand.get();
        if (!isa<RankedTensorType>(oldOperand.getType()))
          continue;
        newOp->setOperand(operand.getOperandNumber(),
                          getValueAs(oldOperand, encoding));
      }
      auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
      newOp->getResult(0).setType(tensorType.cloneWithEncoding(encoding));
      map(op->getResult(0), newOp->getResult(0));
      return newOp;
    }
  }
#endif // __FLAGTREE_RLC_ENHANCE__
  if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp, GatherOp,
#ifdef __FLAGTREE_RLC_ENHANCE__
          ConvertLayoutOp, nvidia_gpu::WarpGroupDotWaitOp>(op) ||
      (hasLayoutPropagationExtensions() && isEncodingUniformPureOp(op))) {
#else  // __FLAGTREE_RLC_ENHANCE__
          ConvertLayoutOp, nvidia_gpu::WarpGroupDotWaitOp>(op)) {
#endif // __FLAGTREE_RLC_ENHANCE__
    Operation *newOp = cloneElementwise(rewriter, op, encoding);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      if (oldResult.getType() == newResult.getType()) {
        oldResult.replaceAllUsesWith(newResult);
        continue;
      }
      map(oldResult, newResult);
    }
    return newOp;
  }
  llvm::report_fatal_error("unexpected op in rewrite");
  return nullptr;
}

bool canBeRemat(Operation *op) {
  if (isa<LoadOp, StoreOp>(op))
    return !isExpensiveLoadOrStore(op);
  if (isa<AtomicRMWOp, AtomicCASOp, DotOp>(op))
    return false;
  if (auto gather = dyn_cast<GatherOp>(op))
    return !gather.getEfficientLayout();

  if (isa<scf::WhileOp, scf::ConditionOp>(op))
    return false;

  return true;
}

#ifdef __FLAGTREE_RLC_ENHANCE__
// A coalesced ("expensive") load normally blocks rematerialization, except a
// vector-sized load (<= one 128-bit fragment per thread): the duplicate is one
// extra L1-resident load, cheaper than the convert's smem round trip. This is
// the per-row scale/bias/mask vector broadcast onto an accumulator. The
// duplicate must keep the access pattern (see isOrderIndifferentAccess).
static bool isDuplicableLoadInEncoding(Operation *op, Attribute encoding,
                                       ModuleAxisInfoAnalysis *axisInfo) {
  auto loadOp = dyn_cast<LoadOp>(op);
  if (!loadOp || !encoding)
    return false;
#ifdef __TLE__
  // A load from a TLE cluster remote pointer lowers to a vectorized
  // shared::cluster load; duplicating or retagging it corrupts that access, so
  // it is never duplicable (see valueOnTleRemotePointerPath).
  if (valueOnTleRemotePointerPath(loadOp.getPtr()))
    return false;
#endif // __TLE__
  auto ptrType = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
  auto resultType = dyn_cast<RankedTensorType>(loadOp.getType());
  if (!ptrType || !resultType)
    return false;
  Type elementType = resultType.getElementType();
  if (!elementType.isIntOrFloat())
    return false;

  // Duplicating pays off only when it removes an expensive convert: a row
  // broadcast onto an MMA accumulator, which costs a shared memory round trip.
  // A blocked-slice (FMA accumulator) convert is cheap, so replicating the load
  // across its warps costs more than it saves. Reject blocked-slice targets.
  if (isSliceEncoding(encoding) && !isMmaSliceEncoding(encoding))
    return false;

  auto mod = op->getParentOfType<ModuleOp>();
  int64_t numThreads =
      lookupNumWarps(op) * TritonGPUDialect::getThreadsPerWarp(mod);
  int64_t elemsPerThread =
      (resultType.getNumElements() + numThreads - 1) / numThreads;
  if (elemsPerThread * elementType.getIntOrFloatBitWidth() > 128)
    return false;

  // The consumer encoding may replicate the value across warps with many
  // elements per thread. A loop-resident load pays that footprint every
  // iteration (where the convert's smem broadcast is cheaper), so cap it.
  bool inLoop = op->getParentOfType<scf::ForOp>() != nullptr ||
                op->getParentOfType<scf::WhileOp>() != nullptr;
  if (inLoop) {
    auto targetType = resultType.cloneWithEncoding(encoding);
    if (int64_t(getTotalElemsPerThread(targetType)) *
            elementType.getIntOrFloatBitWidth() >
        256)
      return false;
  }

  if (preservesWritebackMemoryAccess(ptrType, resultType, encoding,
                                     /*allowNarrowerContiguity=*/false))
    return true;
  // The order-indifference relaxation is limited to straight-line code: for a
  // loop-resident scatter load the retag reshapes every iteration's request
  // stream together with the loop-carried accumulator, degrading the access.
  return !inLoop && isOrderIndifferentAccess(axisInfo, loadOp.getPtr());
}

// A remat must eliminate a layout boundary, not relocate it. If the convert's
// result reaches another convert through layout-transparent ops (view/shape and
// single-tensor-operand elementwise), its target encoding is an intermediate
// stop and committing into it keeps the convert count unchanged; reject such
// roots so the downstream convert absorbs the chain. An op combining the value
// with other tensors genuinely consumes the encoding and ends the walk.
static bool convertResultReachesAnotherConvert(ConvertLayoutOp convertOp) {
  constexpr int kMaxForwardDepth = 4;
  auto isLayoutTransparentUser = [](Operation *op) {
    if (isa<ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp>(op))
      return true;
    if (!op->hasTrait<OpTrait::Elementwise>() || op->getNumRegions() != 0)
      return false;
    int tensorOperands = llvm::count_if(op->getOperands(), [](Value operand) {
      return isa<RankedTensorType>(operand.getType());
    });
    return tensorOperands == 1;
  };

  SmallVector<std::pair<Value, int>> queue{{convertOp.getResult(), 0}};
  while (!queue.empty()) {
    auto [value, depth] = queue.pop_back_val();
    if (depth >= kMaxForwardDepth)
      continue;
    for (Operation *user : value.getUsers()) {
      if (isa<ConvertLayoutOp>(user))
        return true;
      if (!isLayoutTransparentUser(user))
        continue;
      for (Value result : user->getResults())
        if (isa<RankedTensorType>(result.getType()))
          queue.push_back({result, depth + 1});
    }
  }
  return false;
}
#endif // __FLAGTREE_RLC_ENHANCE__

void LayoutRematerialization::updateRematMapping(
    SmallVector<std::tuple<Value, Value>> &values) {
  for (auto [old, newV] : values) {
    auto it = mappedValues.find(old);
    if (it != mappedValues.end()) {
      Attribute encoding = it->second;
      auto rematIt = rematMapping.find({old, it->second});
      assert(rematIt != rematMapping.end());
      Value replacedValue = rematIt->second;
      rematMapping.erase(rematIt);
      mappedValues.erase(it);
      // Loop through the replacement value to find the new version of remat
      // value. This should be okay as the number of values should be small.
      for (auto [before, after] : values) {
        if (before == replacedValue) {
          replacedValue = after;
          break;
        }
      }
      rematMapping[{newV, encoding}] = replacedValue;
      mappedValues[newV] = encoding;
    }
  }
}

void LayoutRematerialization::rewriteSlice(SetVector<Value> &slice,
                                           DenseMap<Value, Attribute> &layout,
                                           ConvertLayoutOp convertOp,
                                           IRMapping &mapping) {
  SetVector<Operation *> opsToRewrite;
  // Keep track of yield operands that need to be duplicated.
  DenseMap<Operation *, SmallVector<int>> yieldOperandsMap;
  // Keep these around to remove them from the slice after our collection pass
  // This ensures we don't duplicate them during an for rewrite or causing the
  // for/yield to fall out of sync
  SetVector<Value> valuesWithExistingRemat;
  for (Value v : slice) {
    auto layoutIt = layout.find(v);
    assert(layoutIt != layout.end());
    // If we already have a remat value for this value, use it.
    if (Value remat = getRematValue(v, layoutIt->second)) {
      mapping.map(v, remat);
      valuesWithExistingRemat.insert(v);
      continue;
    }
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
      if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
        unsigned operandIdx = cast<OpResult>(v).getResultNumber();
        opsToRewrite.insert(ifOp.thenYield().getOperation());
        yieldOperandsMap[ifOp.thenYield()].push_back(operandIdx);
        opsToRewrite.insert(ifOp.elseYield().getOperation());
        yieldOperandsMap[ifOp.elseYield()].push_back(operandIdx);
      }
    } else {
      BlockArgument blockArg = cast<BlockArgument>(v);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = cast<LoopLikeOpInterface>(parentOp)) {
        opsToRewrite.insert(loopOp.getOperation());
        OpOperand *operand = loopOp.getTiedLoopYieldedValue(blockArg);
        auto yieldOp = blockArg.getOwner()->getTerminator();
        yieldOperandsMap[yieldOp].push_back(operand->getOperandNumber());
        opsToRewrite.insert(yieldOp);
      }
    }
  }
  slice.set_subtract(valuesWithExistingRemat);
  opsToRewrite = mlir::topologicalSort(opsToRewrite);

  // replaceAllUsesWith calls delayed until after initial rewrite.
  // This is required for slice.count(value) to work mid rewrite.
  SmallVector<std::tuple<Value, Value>> replacements;

  SmallVector<Operation *> deadOps;
  IRRewriter builder(slice.begin()->getContext());
  for (Operation *op : opsToRewrite) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Keep a mapping of the operands index to the new operands index.
      SmallVector<std::pair<size_t, size_t>> argMapping;
      SmallVector<Value> newOperands;
      for (auto arg : forOp.getRegionIterArgs()) {
        if (slice.count(arg)) {
          OpOperand &initVal = *forOp.getTiedLoopInit(arg);
          argMapping.push_back(std::make_pair(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size()));
          newOperands.push_back(mapping.lookup(initVal.get()));
        }
      }
      // Create a new for loop with the new operands.
      scf::ForOp newForOp = replaceForOpWithNewSignature(
          builder, forOp, newOperands, replacements);
      deadOps.push_back(forOp.getOperation());
      Block &loopBody = *newForOp.getBody();
      for (auto m : argMapping) {
        mapping.map(forOp.getResult(m.first), newForOp.getResult(m.second));
        int numIndVars = newForOp.getNumInductionVars();
        mapping.map(loopBody.getArgument(m.first + numIndVars),
                    loopBody.getArgument(m.second + numIndVars));
        LLVM_DEBUG({
          DBGS() << "mapping forOp "
                 << loopBody.getArgument(m.first + numIndVars) << " to "
                 << loopBody.getArgument(m.second + numIndVars) << '\n';
        });
        // The result is not in the layout/slice, the argument is.
        Value oldArg = loopBody.getArgument(m.first + numIndVars);
        addRematValue(newForOp.getResult(m.first), layout[oldArg],
                      newForOp.getResult(m.second));
        addRematValue(oldArg, layout[oldArg],
                      loopBody.getArgument(m.second + numIndVars));
      }
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> newTypes;
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          auto it = layout.find(res);
          assert(it != layout.end());

          auto oldType = cast<RankedTensorType>(res.getType());
          auto newType = oldType.cloneWithEncoding(it->second);
          newTypes.push_back(newType);
        }
      }
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(builder, ifOp, newTypes, replacements);
      unsigned oldIdx = 0;
      unsigned newIdx = ifOp.getNumResults();
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          // Why can't we use res instead of ifOp.getResult(oldIdx)?
          mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
          addRematValue(ifOp.getResult(oldIdx), layout[res],
                        newIfOp.getResult(newIdx));
          ++newIdx;
        }
        ++oldIdx;
      }
      deadOps.push_back(ifOp.getOperation());
      continue;
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
      SmallVector<int> operandsToRewrite = yieldOperandsMap[op];
      // Sort so that operands are added in the same order as the new scf
      // results/arguments.
      std::sort(operandsToRewrite.begin(), operandsToRewrite.end());
      for (int operandIdx : operandsToRewrite) {
        yieldOperands.push_back(mapping.lookup(yieldOp.getOperand(operandIdx)));
      }
      scf::YieldOp::create(builder, op->getLoc(), yieldOperands);
      op->erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      Operation *newOp = builder.clone(*op);
      auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
      auto newType = tensorType.cloneWithEncoding(layout[op->getResult(0)]);
      auto cvt = ConvertLayoutOp::create(builder, op->getLoc(), newType,
                                         newOp->getResult(0));
      mapping.map(op->getResult(0), cvt.getResult());
      addRematValue(op->getResult(0), layout[op->getResult(0)],
                    cvt.getResult());
      continue;
    }
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end())
        continue;
      auto newType =
          cast<RankedTensorType>(old.getType()).cloneWithEncoding(it->second);
      newV.setType(newType);
      addRematValue(old, it->second, newV);
    }
  }
  // Check mapping and see if there are existing convertOps on the old Argument
  convertOp.replaceAllUsesWith(mapping.lookup(convertOp.getSrc()));
  opToDelete.insert(convertOp);

  updateRematMapping(replacements);
  for (auto &kv : replacements) {
    builder.replaceAllUsesWith(std::get<0>(kv), std::get<1>(kv));
  }

  for (Operation *op : deadOps)
    opToDelete.insert(op);
}

void LayoutRematerialization::rewriteSlice(SetVector<Value> &slice,
                                           DenseMap<Value, Attribute> &layout,
                                           ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, layout, convertOp, mapping);
}

LogicalResult LayoutRematerialization::getConvertBackwardSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation) {
  // Allow re-using existing conversions for a value. Check dominance of any
  // reusable materializations against the root value. This is sufficient
  // because the conversions are processed in post-order.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = getRematValue(value.get(), encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (domInfo.properlyDominates(remat, user)) {
      return remat;
    }
    // FIXME: If the current user is a conversion, then we know it will become
    // a no-op when its operand is replaced with `remat`, but we need to check
    // that its users are all dominated by `remat` so the IR is valid.
    // if (isa<ConvertLayoutOp>(user) && remat.getDefiningOp() &&
    //     domInfo.properlyDominates(user, remat.getDefiningOp())) {
    //   for (Operation *op : user->getUsers()) {
    //     if (!domInfo.dominates(remat, op))
    //       return Value();
    //   }
    //   return remat;
    // }
    return Value();
  };

  return mlir::getConvertBackwardSlice(root, slice, rootEncoding, layout,
                                       stopPropagation, getExistingConversion);
}

LogicalResult LayoutRematerialization::getRematerializableSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
#ifdef __FLAGTREE_RLC_ENHANCE__
    std::function<bool(Operation *)> stopPropagation,
    bool allowDuplicableLoads) {
#else  // __FLAGTREE_RLC_ENHANCE__
    std::function<bool(Operation *)> stopPropagation) {
#endif // __FLAGTREE_RLC_ENHANCE__
  LogicalResult result = getConvertBackwardSlice(root, rootEncoding, slice,
                                                 layout, stopPropagation);
  if (result.failed() || slice.empty())
    return failure();

  // Check if all the operations in the slice can be rematerialized.
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
#ifdef __FLAGTREE_RLC_ENHANCE__
      if (canBeRemat(op))
        continue;
      if (allowDuplicableLoads &&
          isDuplicableLoadInEncoding(op, layout.lookup(v), axisInfoAnalysis))
        continue;
      return failure();
    }
  }

  // A duplicable-load remat must not cross a tt.dot that downgrades the
  // accumulator's register tiling: retagging the accumulator into the small
  // store layout cripples the tile. The intended use (a row broadcast onto an
  // accumulator) never retags the dot. Scoped to the duplicable-load path since
  // otherwise the slice already fails at the dot's input loads.
  if (allowDuplicableLoads) {
    for (Value v : slice) {
      Operation *op = v.getDefiningOp();
      if (!op || !isa<mlir::triton::DotOpInterface>(op))
        continue;
      auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      Attribute newEncoding = layout.lookup(op->getResult(0));
      if (!resultType || !resultType.getEncoding() || !newEncoding)
        continue;
      ArrayRef<int64_t> shape = resultType.getShape();
      if (getTotalElemsPerThread(newEncoding, shape) <
          getTotalElemsPerThread(resultType.getEncoding(), shape))
#else  // __FLAGTREE_RLC_ENHANCE__
      if (!canBeRemat(op))
#endif // __FLAGTREE_RLC_ENHANCE__
        return failure();
    }
  }
  return success();
}

bool LayoutRematerialization::backwardRematerialization() {
  bool changed = false;
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    backwardRematerialization(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    } else {
      changed = true;
    }
  }
  return changed;
}

void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertOnTopOfExtOrBroadcast(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertIntoConditionals() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertIntoConditionals(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

static bool isExpensiveMathOp(Operation *op) {
  // These operations are either multiple instructions or have throughput
  // lower than 16 according to the arithmetic instructions table in:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
  return isa<arith::DivFOp, math::ErfcOp, math::SinhOp, math::CoshOp,
             math::TanhOp, math::AsinhOp, math::AcoshOp, math::AtanhOp,
             math::CtPopOp, math::CountLeadingZerosOp,
             math::CountTrailingZerosOp, math::ExpOp, math::Exp2Op,
             math::ExpM1Op, math::LogOp, math::Log2Op, math::Log10Op,
             math::Log1pOp, math::SinOp, math::CosOp, math::TanOp, math::AsinOp,
             math::AcosOp, math::AtanOp, math::Atan2Op, math::PowFOp,
             math::SqrtOp, math::RsqrtOp, math::ErfOp, math::CbrtOp>(op);
}

static int64_t getByteCount(Value result, int64_t minElementCount = 0,
                            int64_t minBitWidth = 0) {
  int64_t elementCount = 0;
  int64_t dtypeBitWidth = 0;
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    elementCount = tensorTy.getNumElements();
    auto elemType = tensorTy.getElementType();
    if (elemType.isIntOrFloat()) {
      dtypeBitWidth = elemType.getIntOrFloatBitWidth();
    }
  }
  if (elementCount < minElementCount) {
    elementCount = minElementCount;
  }
  if (dtypeBitWidth < minBitWidth) {
    dtypeBitWidth = minBitWidth;
  }
  return (elementCount * dtypeBitWidth) >> 3;
}

void LayoutRematerialization::backwardRematerialization(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return;
  Value oldV = convertOp.getSrc();
  LDBG("check backward remat with source " << oldV << " encoding "
                                           << targetType.getEncoding());
  // Check to see if there are existing remat'ed values for the pair of oldValue
  // and encoding. Make sure it dominates the current conversion.
  Value newV = getRematValue(oldV, targetType.getEncoding());
  if (newV && domInfo.properlyDominates(newV, convertOp)) {
    // Replace it with the remat'ed value.
    convertOp.replaceAllUsesWith(newV);
    opToDelete.insert(convertOp);
    LDBG("found remat'ed value" << newV);
    return;
  }

#ifdef __FLAGTREE_RLC_ENHANCE__
  // 1. Take a backward slice of all rematerializable tensor dependencies. Small
  // access-preserving vector loads may be duplicated (Phase 3), but only when
  // this convert is a final layout boundary; otherwise the duplication would
  // merely move the boundary, so the chain is left for the downstream convert.
  bool allowDuplicableLoads = enableDuplicableLoadRemat &&
                              !convertResultReachesAnotherConvert(convertOp);
#else  // __FLAGTREE_RLC_ENHANCE__
  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
#endif // __FLAGTREE_RLC_ENHANCE__
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
#ifdef __FLAGTREE_RLC_ENHANCE__
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      /*stopPropagation=*/nullptr, allowDuplicableLoads);
#else  // __FLAGTREE_RLC_ENHANCE__
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout);
#endif // __FLAGTREE_RLC_ENHANCE__
  if (result.failed()) {
    LDBG("  getRematerializableSlice failed");
    return;
  }

  // 2. Determine whether rematerialisation is beneficial.

  // Identify all operations in the slice
  SetVector<Operation *> sliceOps;
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      sliceOps.insert(op);
    }
  }

  // Compute single-use operations
  DenseMap<Operation *, bool> isSingleUse;
  std::function<bool(Operation *)> isOpSingleUse;
  isOpSingleUse = [&](Operation *op) -> bool {
    // lookup in memoization array:
    auto it = isSingleUse.find(op);
    if (it != isSingleUse.end()) {
      return it->second;
    }

    bool singleUse = true;

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user == convertOp) {
          continue;
        }
        if (sliceOps.contains(user)) {
          if (!isOpSingleUse(user)) {
            singleUse = false;
            break;
          }
        } else {
          singleUse = false;
          break;
        }
      }
      if (!singleUse) {
        break;
      }
    }

    // insert into memoization array:
    isSingleUse[op] = singleUse;
    return singleUse;
  };

  // Measure the number of bytes that we're manipulating with the
  // ConvertLayoutOp. We pessimistically assume that we round-trip
  // through shared memory and that we cannot vectorise sub-register
  // loads/stores, so we set a minimum element count of 32 (the warp
  // size and number of shared memory banks) and minimum bitwidth of
  // 32 (the width per bank of the shared memory load/store unit).
  int64_t convertLayoutBytes = getByteCount(convertOp.getSrc(), 32, 32);

  // We measure costs in standardised milli-SM-cycles. The smem load
  // and store each cost 8 * convertLayoutBytes, and then we double
  // it to account for extra cost due to synchronisation.
  int64_t convertLayoutCost = 32 * convertLayoutBytes;
  int64_t rematerialisationCost = 0;

  // Evaluate single-use status for every operation in slice
  for (Operation *op : sliceOps) {
    auto dialect = op->getDialect();
    if (isOpSingleUse(op)) {
      // when we rematerialise, this operation does not get duplicated
      // so it does not contribute to our cost model:
      continue;
    } else if (isa<arith::ConstantOp>(op)) {
      // special-case: arith.constant has zero cost
      continue;
    } else if (isa<LoadOp>(op) || isa<LocalLoadOp>(op)) {
      // optimistically assume L1-cached:
      for (Value result : op->getResults()) {
        rematerialisationCost += 8 * getByteCount(result);
      }
    } else if (isa<arith::ArithDialect, math::MathDialect>(dialect)) {
      // this is an arithmetic operation; we distinguish between cheap
      // operations (such as floating point add/mul which can be fused
      // as halves of a single-cycle FMA instruction) and expensive
      // operations which use the special function unit and/or involve
      // multiple instructions.
      int64_t multiplier = isExpensiveMathOp(op) ? 8 : 1;
      for (Value result : op->getResults()) {
        rematerialisationCost += multiplier * getByteCount(result);
      }
    } else if (isa<ReduceOp>(op)) {
      // Reduce op introduce much cost.
      auto reduceOp = dyn_cast<ReduceOp>(op);
      ReduceOpHelper helper(reduceOp);
      if (!helper.isAssociative()) {
        // We shouldn't rematerize a no associative reduce op if it has multiple
        // use chain.
        LDBG("  skipped rematerialization due to non-associative reduce in the "
             "slice");
        return;
      }
      rematerialisationCost += helper.getIntraWarpSizeWithUniqueData();
      rematerialisationCost += 8 * helper.getInterWarpSizeWithUniqueData();
    }
  }

  LLVM_DEBUG({
    DBGS() << "  convert layout cost: " << convertLayoutCost << "\n";
    DBGS() << "  rematerialisation cost: " << rematerialisationCost << "\n";
  });

  if (rematerialisationCost > convertLayoutCost) {
    LDBG("  skipped rematerialization due to higher cost");
    return;
  }

  LLVM_DEBUG({
    DBGS() << "  remat convert op " << convertOp << '\n';
    for (Value v : slice)
      DBGS() << "    " << v << '\n';
  });

  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp);
}

#ifdef __FLAGTREE_RLC_ENHANCE__
// -----------------------------------------------------------------------------
// Phase 3: store-layout rematerialization
// -----------------------------------------------------------------------------
//
// A global writeback whose value comes from a convert_layout forces a
// conversion right before the write, e.g. the matmul writeback
//
//   %acc  = tt.dot ...                 : tensor<...x#mma>
//   %cvt  = ttg.convert_layout %acc    : #mma -> #blocked   (round-trips smem)
//   tt.store %ptr, %cvt, %mask         : #blocked
//
// Backward remat cannot remove it because the source (the dot accumulator) is a
// frozen anchor. Instead we rematerialize the cheap address/mask chains in the
// value's layout and write the value as-is:
//
//   %ptr2  = <addr chain in #mma>
//   %mask2 = <mask chain in #mma>
//   tt.store %ptr2, %acc, %mask2       : #mma
//
// The same applies to a tt.atomic_rmw with unused result (e.g. a split-k
// epilogue). Only attempted when the target layout preserves the memory access
// pattern and the address/mask chains are pure index/mask math (no loads,
// reductions, anchors or block arguments).

// A global writeback shares the (ptr, value, mask?) operand layout between
// tt.store and tt.atomic_rmw, and requires all its tensor operands to use one
// encoding. An atomic qualifies only when its result is unused, so changing
// the result encoding has no consumers to disturb.
static bool isGlobalWritebackOp(Operation *op) {
  if (isa<StoreOp>(op))
    return true;
  if (auto atomicOp = dyn_cast<AtomicRMWOp>(op))
    return atomicOp.getResult().use_empty();
  return false;
}

bool LayoutRematerialization::rematerializeWritebackLayout(
    Operation *writebackOp) {
  // The written value must be produced by a convert_layout, otherwise there is
  // no writeback conversion to eliminate.
  OpOperand &ptrOperand = writebackOp->getOpOperand(0);
  OpOperand &valueOperand = writebackOp->getOpOperand(1);
  auto valueConvert = valueOperand.get().getDefiningOp<ConvertLayoutOp>();
  if (!valueConvert)
    return false;
  // The conversion must feed this writeback exclusively; otherwise it stays
  // live for its other users and we gain nothing.
  if (!hasSingleUse(valueConvert.getResult()))
    return false;

  auto ptrType = dyn_cast<RankedTensorType>(ptrOperand.get().getType());
  auto valueType = dyn_cast<RankedTensorType>(valueOperand.get().getType());
  auto targetType = dyn_cast<RankedTensorType>(valueConvert.getSrc().getType());
  if (!ptrType || !valueType || !targetType)
    return false;
  Attribute targetEncoding = targetType.getEncoding();
  if (!targetEncoding || targetEncoding == valueType.getEncoding())
    return false;

  // Only target tensor-core (MMA) value layouts. These are the layouts whose
  // conversion is expensive (it round-trips through shared memory) and that
  // the earlier blocked-to-blocked phases intentionally leave untouched.
  if (!isa<MmaEncodingTrait>(targetEncoding))
    return false;
  // Trading per-thread store width for the removed convert only pays off on a
  // hot path: a writeback inside a loop (the convert serializes every
  // iteration) or an atomic (element-granular anyway). One-shot epilogues must
  // keep the store's contiguity, else the decoalesced write costs more.
  bool allowNarrowerContiguity = isa<AtomicRMWOp>(writebackOp) ||
                                 writebackOp->getParentOfType<scf::ForOp>() ||
                                 writebackOp->getParentOfType<scf::WhileOp>();
  if (!preservesWritebackMemoryAccess(ptrType, valueType, targetEncoding,
                                      allowNarrowerContiguity))
    return false;

  // A contiguous (coalesced) store moves consecutive addresses across a warp
  // even at sizePerThread=1, so the contiguity comparison above understates its
  // bandwidth; retagging it onto the fragmented MMA layout breaks that warp
  // coalescing. Only pays off with no coalescing to lose (a scatter store, or
  // an atomic). Guards a fully-coalesced persistent-matmul C store the loop
  // exemption above would otherwise admit.
  if (!isa<AtomicRMWOp>(writebackOp) && axisInfoAnalysis &&
      !isOrderIndifferentAccess(axisInfoAnalysis, ptrOperand.get()))
    return false;

  // The address (and optional mask) chains must be cheaply rematerializable in
  // the value's layout. Requiring pure index/mask math guarantees that moving
  // the writeback into the value's layout does not duplicate loads/reductions
  // or disturb a layout anchor.
  auto collectWritebackOperandSlice =
      [&](OpOperand &operand, SetVector<Value> &slice,
          DenseMap<Value, Attribute> &layout) -> bool {
    if (!isa<RankedTensorType>(operand.get().getType()))
      return true; // Scalar operand: nothing to rematerialize.
    if (failed(
            getRematerializableSlice(operand, targetEncoding, slice, layout)))
      return false;
    for (Value v : slice) {
      if (isa<BlockArgument>(v))
        return false;
      Operation *def = v.getDefiningOp();
      if (!def || def->getNumRegions() != 0 || isLayoutAnchor(def) ||
          isa<LoadOp, LocalLoadOp, ReduceOp, ConvertLayoutOp>(def))
        return false;
    }
    return true;
  };

  // The writeback operands are [ptr, value, mask?]; the mask is optional.
  OpOperand *maskOperand = writebackOp->getNumOperands() > 2
                               ? &writebackOp->getOpOperand(2)
                               : nullptr;

  SetVector<Value> ptrSlice, maskSlice;
  DenseMap<Value, Attribute> ptrLayout, maskLayout;
  if (!collectWritebackOperandSlice(ptrOperand, ptrSlice, ptrLayout))
    return false;
  if (maskOperand &&
      !collectWritebackOperandSlice(*maskOperand, maskSlice, maskLayout))
    return false;

  // All checks passed: commit. Insert a temporary convert on each address/mask
  // operand, then reuse the existing slice-rewrite machinery to push the
  // conversion up into the (cheap) producer chain and drop it.
  OpBuilder builder(writebackOp);
  auto rematerializeOperand = [&](OpOperand &operand, SetVector<Value> &slice,
                                  DenseMap<Value, Attribute> &layout) {
    if (slice.empty())
      return;
    auto operandType = cast<RankedTensorType>(operand.get().getType());
    auto newType = operandType.cloneWithEncoding(targetEncoding);
    auto tmpConvert = ConvertLayoutOp::create(builder, writebackOp->getLoc(),
                                              newType, operand.get());
    operand.set(tmpConvert.getResult());
    rewriteSlice(slice, layout, tmpConvert);
  };
  rematerializeOperand(ptrOperand, ptrSlice, ptrLayout);
  if (maskOperand)
    rematerializeOperand(*maskOperand, maskSlice, maskLayout);

  // The value can now be written directly in its own layout. An atomic's
  // (unused) result must follow the value encoding.
  valueOperand.set(valueConvert.getSrc());
  if (auto atomicOp = dyn_cast<AtomicRMWOp>(writebackOp))
    atomicOp.getResult().setType(targetType);
  if (valueConvert.getResult().use_empty())
    opToDelete.insert(valueConvert);
  return true;
}

bool LayoutRematerialization::rematerializeStoreLayout() {
  bool changed = false;
  SmallVector<Operation *> writebacks;
  funcOp.walk([&](Operation *op) {
    if (isGlobalWritebackOp(op))
      writebacks.push_back(op);
  });
  for (Operation *op : writebacks)
    changed |= rematerializeWritebackLayout(op);
  return changed;
}

bool LayoutRematerialization::rematerializeLocalStoreLayout(
    LocalStoreOp storeOp) {
  auto valueConvert = storeOp.getSrc().getDefiningOp<ConvertLayoutOp>();
  if (!valueConvert || !hasSingleUse(valueConvert.getResult()))
    return false;

  auto targetType = dyn_cast<RankedTensorType>(valueConvert.getSrc().getType());
  if (!targetType || !targetType.getEncoding())
    return false;

  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  if (failed(getRematerializableSlice(storeOp.getSrcMutable(),
                                      targetType.getEncoding(), slice, layout)))
    return false;
  if (slice.empty())
    return false;

  for (Value value : slice) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp || isLayoutAnchor(defOp) ||
        isa<LocalLoadOp, LocalStoreOp, ReduceOp, ConvertLayoutOp>(defOp))
      return false;
  }

  OpBuilder builder(storeOp);
  auto storeSrcType = cast<RankedTensorType>(storeOp.getSrc().getType());
  auto tmpType = storeSrcType.cloneWithEncoding(targetType.getEncoding());
  auto tmpConvert = ConvertLayoutOp::create(builder, storeOp.getLoc(), tmpType,
                                            storeOp.getSrc());
  storeOp.getSrcMutable().assign(tmpConvert.getResult());
  rewriteSlice(slice, layout, tmpConvert);
  return true;
}

bool LayoutRematerialization::rematerializeLocalStoreLayout() {
  bool changed = false;
  SmallVector<LocalStoreOp> stores;
  funcOp.walk([&](LocalStoreOp storeOp) { stores.push_back(storeOp); });
  for (LocalStoreOp storeOp : stores)
    changed |= rematerializeLocalStoreLayout(storeOp);
  return changed;
}
#endif // __FLAGTREE_RLC_ENHANCE__

void LayoutRematerialization::hoistConvertDotOperand() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertDotOperand(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertDotOperand(
    ConvertLayoutOp convertOp) {
  auto targetType = convertOp.getType();
  // The pass is targeted to MMA dot operands

#ifdef __TLE__
  {
    DenseSet<Value> visited;
    if (touchesTleRemotePointerPath(convertOp.getSrc(), visited))
      return;
  }
#endif // __TLE__

  auto canBePipelined = [&](ConvertLayoutOp convertOp) {
    // FIXME: Check that the parent is a for loop
    auto parent = convertOp->getParentOp();
    if (!parent)
      return false;

    // Find all the dot-like ops in the for loop that have a dot operand
    // encoding on the lhs and check if any of them post-dominates the load +
    // cvt
    SmallVector<Operation *> dotLikeOps;
    parent->walk([&](Operation *op) {
      if (!isa<mlir::triton::DotOpInterface>(op))
        return;
      auto opType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!opType)
        return;
      auto dotEnc = dyn_cast<DotOperandEncodingAttr>(opType.getEncoding());
      if (!dotEnc)
        return;
      if (isa<MmaEncodingTrait>(dotEnc.getParent()))
        dotLikeOps.push_back(op);
    });
    if (dotLikeOps.empty())
      return false;
    return llvm::any_of(dotLikeOps, [&](Operation *dot) {
      return postDomInfo.postDominates(dot, convertOp);
    });
  };

  // We move convert #dot_operand next to their loads. This is done
  // so that it's then easy to pipeline these loads
  if (!canBePipelined(convertOp))
    return;

  // We hoist over any operation that can be done without data movement between
  // threads We do views and elementwise pure ops for now
  auto noDataMovement = [](Operation *op) {
    return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
           isa<BroadcastOp, Fp4ToFpOp, ConvertLayoutOp, UpcastFpOpInterface>(
               op) ||
           isView(op);
  };
  // Stop the slice as soon as we find an operation that cannot be done without
  // data movement between threads
  auto stop = std::not_fn(noDataMovement);

  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  // Set-up the conversion "cache"
  LogicalResult result = getConvertBackwardSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout, stop);
  if (result.failed())
    return;

  IRMapping mapping;
  OpBuilder builder(convertOp.getContext());
  SetVector<Value> innerSlice;
  for (Value v : slice) {
    if (!v.getDefiningOp()) {
      LLVM_DEBUG(
          { DBGS() << "  Block arguments not supported. Got " << v << "\n"; });
      return;
    }

    // We expect the leaves of the slice to be Load, DescriptorLoad or
    // arith::Constant This could be generalised if necessary
    if (!isa<LoadOp, DescriptorLoadOp>(v.getDefiningOp())) {
      auto op = v.getDefiningOp();
      if (isa<arith::ConstantOp>(op) || noDataMovement(op)) {
        innerSlice.insert(v);
        continue;
      } else {
        LLVM_DEBUG({
          DBGS() << "  Leaves must be Load, DescriptorLoad or Constant. Got "
                 << v << "\n";
        });
        return;
      }
    }
    Operation *loadOp = v.getDefiningOp();
    builder.setInsertionPointAfter(loadOp);
    auto type = dyn_cast<RankedTensorType>(loadOp->getResult(0).getType());
    if (!type)
      continue;
    auto newType = type.cloneWithEncoding(layout[loadOp->getResult(0)]);
    auto newConvertOp = ConvertLayoutOp::create(builder, convertOp.getLoc(),
                                                newType, loadOp->getResult(0));
    mapping.map(loadOp->getResult(0), newConvertOp.getResult());
  }

  if (innerSlice.empty()) {
    return;
  }

  LLVM_DEBUG({
    DBGS() << "  Hoisting " << convertOp << '\n';
    for (Value v : innerSlice)
      DBGS() << "    " << v << '\n';
  });

  rewriteSlice(innerSlice, layout, convertOp, mapping);
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return;

  auto isExtOrBroadcastOp = [](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, BroadcastOp,
            ExpandDimsOp>(op)) {
      return true;
    }
    if (auto fpToFpOp = dyn_cast<FpToFpOp>(op)) {
      auto srcType = cast<RankedTensorType>(fpToFpOp.getOperand().getType());
      return getElementBitWidth(srcType) <
             getElementBitWidth(cast<RankedTensorType>(fpToFpOp.getType()));
    }
    return false;
  };
  // 1. Take a backward slice of all the tensor dependencies.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      isExtOrBroadcastOp);
  if (result.failed())
    return;

  Operation *extOrBroadcastOp = nullptr;
  unsigned sliceSize = slice.size();
  for (unsigned i = 0; i < sliceSize; i++) {
    Value v = slice[i];
    Operation *op = v.getDefiningOp();
    if (!op)
      continue;
    if (isExtOrBroadcastOp(op)) {
      SetVector<Value> tempSlice;
      DenseMap<Value, Attribute> tempLayout;
      Attribute srcEncoding = inferSrcEncoding(op, layout[v]);
      if (!srcEncoding)
        return;
      LogicalResult result = getRematerializableSlice(
          op->getOpOperand(0), srcEncoding, tempSlice, tempLayout);

      // If a value is already assigned to a _different_ layout,
      // we cannot propagate past this op (as it would conflict with
      // an already-assigned layout).
      for (auto [val, enc] : tempLayout) {
        auto preexistingLayout = layout.find(val);
        if (preexistingLayout != layout.end() &&
            preexistingLayout->second != enc) {
          result = failure();
          break;
        }
      }

      // If we can rematerialize the rest of the ext slice we can ignore this
      // ext as it won't need a convert.
      if (result.succeeded()) {
        slice.insert(tempSlice.begin(), tempSlice.end());
        layout.insert(tempLayout.begin(), tempLayout.end());
        continue;
      }
      // Only apply it if there is a single ext op otherwise we would have to
      // duplicate the convert.
      if (extOrBroadcastOp != nullptr)
        return;
      extOrBroadcastOp = op;
    }
  }

  if (extOrBroadcastOp == nullptr)
    return;
  Attribute dstEncoding = layout[extOrBroadcastOp->getResult(0)];
  Attribute srcEncoding = inferSrcEncoding(extOrBroadcastOp, dstEncoding);
  if (!srcEncoding)
    return;
  // Move the convert before the ext op and rewrite the slice.
  OpBuilder builder(extOrBroadcastOp);
  auto tensorType =
      cast<RankedTensorType>(extOrBroadcastOp->getOperand(0).getType());
  auto newType = tensorType.cloneWithEncoding(srcEncoding);
  auto newConvertOp = ConvertLayoutOp::create(
      builder, convertOp.getLoc(), newType, extOrBroadcastOp->getOperand(0));
  Operation *newExtOrBroadcast = builder.clone(*extOrBroadcastOp);
  newExtOrBroadcast->setOperand(0, newConvertOp.getResult());
  auto oldExtOrBroadcastType =
      cast<RankedTensorType>(extOrBroadcastOp->getResult(0).getType());
  Type newExtOrBroadcastType =
      oldExtOrBroadcastType.cloneWithEncoding(dstEncoding);
  newExtOrBroadcast->getResult(0).setType(newExtOrBroadcastType);
  IRMapping mapping;
  mapping.map(extOrBroadcastOp->getResult(0), newExtOrBroadcast->getResult(0));
  slice.remove(extOrBroadcastOp->getResult(0));
  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp, mapping);
}

void LayoutRematerialization::hoistConvertIntoConditionals(
    ConvertLayoutOp convertOp) {
  // Take the backward slice of tensor dependencies rooted at the conversion,
  // stopping at conditionals. This subslice is used to initialize the analysis.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  auto isIfOp = [](Operation *op) { return isa<scf::IfOp>(op); };
  if (failed(getRematerializableSlice(convertOp.getSrcMutable(),
                                      convertOp.getType().getEncoding(), slice,
                                      layout, isIfOp)))
    return;

  // These are the conditional edges above which conversions should be hoisted.
  // The value represents the `scf.if` op result and the operand represents the
  // edge into one of the branches.
  SmallVector<std::pair<Value, OpOperand *>> hoistAbove;

  // The list of `scf.if` op results in the slice that are not rematerializable.
  // Hoisting is terminated at these values.
  SmallVector<OpResult> terminals;

  // This loop recurses through the subslices of the backwards dependencies, so
  // re-query the size of `slice`.
  for (unsigned i = 0; i != slice.size(); ++i) {
    Value v = slice[i];
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      continue;

    Attribute rootLayout = layout.at(v);
    unsigned resIdx = cast<OpResult>(v).getResultNumber();

    // Take the backward slice along each branch.
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    OpOperand &thenRes = thenYield.getResultsMutable()[resIdx];
    OpOperand &elseRes = elseYield.getResultsMutable()[resIdx];

    SetVector<Value> thenSlice, elseSlice;
    DenseMap<Value, Attribute> thenLayout, elseLayout;

    LogicalResult thenResult = getRematerializableSlice(
        thenRes, rootLayout, thenSlice, thenLayout, isIfOp);
    LogicalResult elseResult = getRematerializableSlice(
        elseRes, rootLayout, elseSlice, elseLayout, isIfOp);

    // If propagation across both edges of this conditional succeeded, then we
    // don't need to hoist across it. Merge into the current slice.
    if (succeeded(thenResult) && succeeded(elseResult)) {
      slice.insert(thenSlice.begin(), thenSlice.end());
      slice.insert(elseSlice.begin(), elseSlice.end());
      layout.insert(thenLayout.begin(), thenLayout.end());
      layout.insert(elseLayout.begin(), elseLayout.end());
      continue;
    }

    // If propagation across both edges failed, then this conditional
    // terminates backwards rematerialization.
    if (failed(thenResult) && failed(elseResult)) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // Only hoist into conditionals inside loops. The assumption is that an if
    // inside a loop executes fewer than the total number of loop iterations,
    // making this hoist profitable.
    if (!isa<scf::ForOp>(ifOp->getParentOp())) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // The layout conversion can be rematerialized along one edge but not the
    // other. We can hoist the conversion into the other branch. Push this
    // into the subslice list for analysis.
    if (succeeded(thenResult)) {
      hoistAbove.emplace_back(v, &elseRes);
      slice.insert(thenSlice.begin(), thenSlice.end());
      layout.insert(thenLayout.begin(), thenLayout.end());
    } else {
      hoistAbove.emplace_back(v, &thenRes);
      slice.insert(elseSlice.begin(), elseSlice.end());
      layout.insert(elseLayout.begin(), elseLayout.end());
    }
  }

  // Exit early if there is nothing to do.
  if (hoistAbove.empty())
    return;

  // Rematerialize failed hoists right before the condtional, and hoist those
  // that succeeded into the branch and then rewrite the slice.
  IRMapping mapping;
  auto hoistRemat = [&](OpBuilder &b, Value v, Attribute encoding) {
    auto tensorType = cast<RankedTensorType>(v.getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    Value newCvt = ConvertLayoutOp::create(b, convertOp.getLoc(), newType, v);

    mapping.map(v, newCvt);
    slice.remove(v);
  };
  for (Value v : terminals) {
    OpBuilder b(v.getContext());
    b.setInsertionPointAfter(v.getDefiningOp());
    hoistRemat(b, v, layout.at(v));
  }
  for (auto [result, edge] : hoistAbove) {
    OpBuilder b(edge->getOwner());
    hoistRemat(b, edge->get(), layout.at(result));
  }
  rewriteSlice(slice, layout, convertOp, mapping);
}

#ifdef __FLAGTREE_RLC_ENHANCE__
bool backwardRematerialization(ModuleOp module, bool duplicableLoadRemat) {
#else  // __FLAGTREE_RLC_ENHANCE__
bool backwardRematerialization(ModuleOp module) {
#endif // __FLAGTREE_RLC_ENHANCE__
  bool changed = false;
#ifdef __FLAGTREE_RLC_ENHANCE__
  // The duplicable-load gate consults real address contiguity. Values created
  // by earlier rewrites are simply absent from the analysis and resolve
  // conservatively.
  std::optional<ModuleAxisInfoAnalysis> axisInfoAnalysis;
  if (duplicableLoadRemat)
    axisInfoAnalysis.emplace(module);
  ModuleAxisInfoAnalysis *axisInfo =
      axisInfoAnalysis ? &*axisInfoAnalysis : nullptr;
#endif // __FLAGTREE_RLC_ENHANCE__
  module.walk([&](FuncOp funcOp) {
#ifdef __FLAGTREE_RLC_ENHANCE__
    LayoutRematerialization layoutRemat(funcOp, duplicableLoadRemat, axisInfo);
#else  // __FLAGTREE_RLC_ENHANCE__
    LayoutRematerialization layoutRemat(funcOp);
#endif // __FLAGTREE_RLC_ENHANCE__
    changed |= layoutRemat.backwardRematerialization();
    layoutRemat.cleanup();
  });
  return changed;
}

#ifdef __FLAGTREE_RLC_ENHANCE__
// Phase 3 driver: rematerialize writeback address/mask chains in the written
// value's layout to drop writeback conversions (see
// rematerializeWritebackLayout and rematerializeLocalStoreLayout).
bool rematerializeStoreLayout(ModuleOp module) {
  bool changed = false;
  // Writeback rematerialization consults the store's real address contiguity to
  // keep coalesced (non-scatter) stores on their own layout; see
  // rematerializeWritebackLayout.
  ModuleAxisInfoAnalysis axisInfoAnalysis(module);
  module.walk([&](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp, /*duplicableLoadRemat=*/false,
                                        &axisInfoAnalysis,
                                        /*storeLayoutRemat=*/true);
    changed |= layoutRemat.rematerializeStoreLayout();
    changed |= layoutRemat.rematerializeLocalStoreLayout();
    layoutRemat.cleanup();
  });
  return changed;
}
#endif // __FLAGTREE_RLC_ENHANCE__

#ifdef __TLE__
static void eraseOpsCreatedAfter(Operation *previous, Operation *insertBefore) {
  Operation *cur =
      previous ? previous->getNextNode() : &insertBefore->getBlock()->front();
  while (cur && cur != insertBefore) {
    Operation *next = cur->getNextNode();
    cur->erase();
    cur = next;
  }
}

bool rematerializeStoreLayoutDemands(ModuleOp module) {
  bool changed = false;
  EncodingRematerializationPolicy policy;
  IRRewriter rewriter(module.getContext());
  DominanceInfo dominance(module);

  SmallVector<StoreOp> stores;
  module.walk([&](StoreOp store) { stores.push_back(store); });

  for (StoreOp store : stores) {
    if (!store || !store->getBlock())
      continue;
    auto valueConvert = store.getValue().getDefiningOp<ConvertLayoutOp>();
    if (!valueConvert)
      continue;
    auto targetTy = dyn_cast<RankedTensorType>(valueConvert.getSrc().getType());
    if (!targetTy || !targetTy.getEncoding())
      continue;
    if (!isa<NvidiaMmaEncodingAttr>(targetTy.getEncoding()))
      continue;

    Operation *rollbackPrevious = store->getPrevNode();
    EncodingRematerializationCache cache;
    auto ptr = rematerializeWithEncoding(rewriter, store, store.getPtr(),
                                         targetTy.getEncoding(), cache,
                                         dominance, policy);
    if (failed(ptr))
      continue;

    Value mask;
    if (Value oldMask = store.getMask()) {
      auto rematMask = rematerializeWithEncoding(rewriter, store, oldMask,
                                                 targetTy.getEncoding(), cache,
                                                 dominance, policy);
      if (failed(rematMask)) {
        eraseOpsCreatedAfter(rollbackPrevious, store);
        continue;
      }
      mask = *rematMask;
    }

    store.getPtrMutable().assign(*ptr);
    store.getValueMutable().assign(valueConvert.getSrc());
    if (mask)
      store.getMaskMutable().assign(mask);
    if (valueConvert->use_empty())
      valueConvert.erase();
    changed = true;
  }

  return changed;
}

FailureOr<Attribute>
inferLocalStoreSourceTargetEncoding(LocalStoreOp store,
                                    DominanceInfo &dominance,
                                    EncodingRematerializationPolicy &policy) {
  auto sourceTy = dyn_cast<RankedTensorType>(store.getSrc().getType());
  if (!sourceTy || !sourceTy.getEncoding())
    return failure();

  if (auto convert = store.getSrc().getDefiningOp<ConvertLayoutOp>()) {
    auto convertSrcTy = dyn_cast<RankedTensorType>(convert.getSrc().getType());
    if (convertSrcTy && convertSrcTy.getEncoding() &&
        isa<NvidiaMmaEncodingAttr>(convertSrcTy.getEncoding()) &&
        convertSrcTy.getEncoding() != sourceTy.getEncoding())
      return convertSrcTy.getEncoding();
  }

  SmallVector<Attribute, 4> equivalentMmaEncodings;
  collectAvailableEquivalentNvidiaMmaEncodings(store.getSrc(),
                                               store.getOperation(), dominance,
                                               policy, equivalentMmaEncodings);
  if (equivalentMmaEncodings.empty())
    return failure();
  if (equivalentMmaEncodings.front() == sourceTy.getEncoding())
    return failure();
  return equivalentMmaEncodings.front();
}

bool rematerializeLocalStoreLayoutDemands(ModuleOp module) {
  bool changed = false;
  EncodingRematerializationPolicy policy;
  IRRewriter rewriter(module.getContext());
  DominanceInfo dominance(module);

  SmallVector<LocalStoreOp> stores;
  module.walk([&](LocalStoreOp store) { stores.push_back(store); });

  for (LocalStoreOp store : stores) {
    if (!store || !store->getBlock())
      continue;

    auto targetEncoding =
        inferLocalStoreSourceTargetEncoding(store, dominance, policy);
    if (failed(targetEncoding))
      continue;

    EncodingRematerializationCache cache;
    auto source =
        rematerializeWithEncoding(rewriter, store, store.getSrc(),
                                  *targetEncoding, cache, dominance, policy);
    if (failed(source))
      continue;

    store.getSrcMutable().assign(*source);
    changed = true;
  }

  return changed;
}
#endif // __TLE__

void hoistConvert(ModuleOp module) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk([](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp);
    layoutRemat.hoistConvertOnTopOfExtOrBroadcast();
    layoutRemat.cleanup();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertIntoConditionals();
    layoutRemat.cleanup();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertDotOperand();
    layoutRemat.cleanup();
  });
}
} // namespace

class TritonGPURemoveLayoutConversionsPass
    : public impl::TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
#ifdef __FLAGTREE_RLC_ENHANCE__
  TritonGPURemoveLayoutConversionsPass() = default;
  explicit TritonGPURemoveLayoutConversionsPass(bool enhance)
      : rlcEnhance(enhance) {}
#endif // __FLAGTREE_RLC_ENHANCE__
  // Cleanup convert ops.
  void cleanupConvertOps() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet cleanUpPatterns(context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
    if (applyPatternsGreedily(m, std::move(cleanUpPatterns)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      DBGS() << "Module after canonicalizing:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

#ifdef __FLAGTREE_RLC_ENHANCE__
    bool costBased = rlcEnhance && kEnableCostBasedResolution;
    bool backwardProp = rlcEnhance && kEnableBackwardPropagation;
    bool smallComponentSolving = rlcEnhance && kEnableSmallComponentSolving;
    bool storeLayoutRemat = rlcEnhance && kEnableStoreLayoutRematerialization;

    if (!costBased) {
      backwardProp = false;
      smallComponentSolving = false;
    }
    // The phase extensions consult the real address pattern of loads (see
    // isOrderIndifferentAccess); the analysis is only built when they
    // are enabled so the all-disabled path stays identical to the original
    // pass.
    std::optional<ModuleAxisInfoAnalysis> axisInfoAnalysis;
    if (smallComponentSolving)
      axisInfoAnalysis.emplace(m);
    ModuleAxisInfoAnalysis *axisInfo =
        axisInfoAnalysis ? &*axisInfoAnalysis : nullptr;
    m.walk([costBased, backwardProp, smallComponentSolving,
            axisInfo](FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp, costBased, backwardProp,
                                          smallComponentSolving, axisInfo);
#else  // __FLAGTREE_RLC_ENHANCE__
    // 1. Propagate layout forward starting from "anchor" ops.
    m.walk([](FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp);
#endif // __FLAGTREE_RLC_ENHANCE__
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
#ifdef __FLAGTREE_RLC_ENHANCE__
      if (layoutPropagation.propagateLayoutBackward())
        layoutPropagation.propagateLayout();
      if (layoutPropagation.solveSmallComponents())
        layoutPropagation.propagateLayout();
#endif // __FLAGTREE_RLC_ENHANCE__
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    LLVM_DEBUG({
#ifdef __FLAGTREE_RLC_ENHANCE__
      DBGS() << "Module after layout propagation:\n";
#else  // __FLAGTREE_RLC_ENHANCE__
      DBGS() << "Module after propagating layouts forward:\n";
#endif // __FLAGTREE_RLC_ENHANCE__
      m.dump();
    });

    cleanupConvertOps();

    bool changed = false;
    do {
      changed = false;
      // 2. For remaining convert ops, try to rematerialize the slice of
#ifdef __FLAGTREE_RLC_ENHANCE__
      // producer operation to avoid having to convert. Under Phase 3 the
      // slices may additionally duplicate small access-preserving vector
      // loads (a per-row vector consumed in two layouts).
      changed = backwardRematerialization(m, storeLayoutRemat);
#else
      // producer operation to avoid having to convert.
      changed = backwardRematerialization(m);
#endif // __FLAGTREE_RLC_ENHANCE__
      LLVM_DEBUG({
        DBGS() << "Module after backward remat:\n";
        m.dump();
      });

      // Cleanup dummy converts created during backward remat.
      cleanupConvertOps();
#ifdef __FLAGTREE_RLC_ENHANCE__
      // 3. Phase 3: drop store writeback conversions by rematerializing the
      // address/mask chains in the stored value's layout (e.g. MMA matmul
      // writeback). Run inside the fixed-point loop so the new layouts feed
      // back into backward remat / cleanup.
      if (storeLayoutRemat) {
        bool storeChanged = rematerializeStoreLayout(m);
        if (storeChanged) {
          changed = true;
          cleanupConvertOps();
        }
        LLVM_DEBUG({
          DBGS() << "Module after store layout remat:\n";
          m.dump();
        });
      }
#endif // __FLAGTREE_RLC_ENHANCE__
#ifdef __TLE__
      if (m->hasAttr(
              ::mlir::triton::tle::kTleEnableEncodingRematerializationAttr)) {
        changed |= rematerializeStoreLayoutDemands(m);
        changed |= rematerializeLocalStoreLayoutDemands(m);
        cleanupConvertOps();
      }
#endif // __TLE__
    } while (changed);
#ifdef __FLAGTREE_RLC_ENHANCE__
    // 4. For remaining converts, try to hoist them above cast generating larger
#else  // __FLAGTREE_RLC_ENHANCE__
    // 3. For remaining converts, try to hoist them above cast generating larger
#endif // __FLAGTREE_RLC_ENHANCE__
    // size types in order to reduce the cost of the convert op.
    hoistConvert(m);
    LLVM_DEBUG({
      DBGS() << "Module after hoisting converts:\n";
      m.dump();
    });

#ifdef __FLAGTREE_RLC_ENHANCE__
    // 5. Apply clean up patterns to remove remove dead convert and dead code
#else  // __FLAGTREE_RLC_ENHANCE__
    // 4. Apply clean up patterns to remove remove dead convert and dead code
#endif // __FLAGTREE_RLC_ENHANCE__
    // generated by the previous transformations.
    RewritePatternSet cleanUpPatterns2(context);
    populateForOpDeadArgumentElimination(cleanUpPatterns2);
    scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    if (applyPatternsGreedily(m, std::move(cleanUpPatterns2)).failed()) {
      signalPassFailure();
    }
    LLVM_DEBUG({
      DBGS() << "Module after final cleanups:\n";
      m.dump();
    });
  }
#ifdef __FLAGTREE_RLC_ENHANCE__
private:
  bool rlcEnhance = false;
#endif // __FLAGTREE_RLC_ENHANCE__
};
#ifdef __FLAGTREE_RLC_ENHANCE__
// FlagTree entry point that injects the runtime master switch. The NVIDIA
// backend calls this with FLAGTREE_RLC_ENHANCE (see passes.cc / compiler.py);
// other callers use the plain 0-arg factory and keep the original behavior.
std::unique_ptr<::mlir::Pass>
createTritonGPURemoveLayoutConversionsEnhanced(bool enhance) {
  return std::make_unique<TritonGPURemoveLayoutConversionsPass>(enhance);
}
#endif // __FLAGTREE_RLC_ENHANCE__

} // namespace mlir::triton::gpu

#endif
