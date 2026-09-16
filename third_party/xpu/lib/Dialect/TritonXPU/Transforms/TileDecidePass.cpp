//===----------------------------------------------------------------------===//
// tritonxpu-tile-decide -- the tiling decision, taken ahead of CoreTiling.
//
// M2/M3 of vectorize&unrollcontrol_design.md §3. Today the whole decision lives
// inside `tritonxpu-unroll-control`: it builds the op tree, measures pressure,
// enumerates legal trip counts and calls `TileDecider` in the same walk that
// rewrites. This pass pulls out the half that does not need E -- which
// representation each root ends in, and therefore where the vector/scalar
// boundaries fall -- and takes it at the position where the states are
// decidable: immediately after the `pre-tiling` vectorizability analysis
// (compiler.py:390), before CoreTiling fixes sizePerCore.
//
// What stays behind, and why it is not a shortcut: tier 1 (VRF budget) and tier
// 2 (LM footprint) both rank candidates by peak pressure over a *fixed*
// per-core geometry. Neither `numCol` nor `widthPerCore` nor any peak exists
// here, so the two hard tiers cannot be evaluated -- they are recorded as
// deferred, and `iterNum` continues to be chosen at compiler.py:433 from the
// real IR. §6 says the same about M3's first version: report, do not decide.
//
// The pass never mutates the IR. Everything it produces goes into the
// `triton_xpu.tile_decision` carrier, which `tritonxpu-tile-analysis` resolves
// and erases before emission. That is deliberate: a decision pass whose only
// output is an attribute can be inserted and removed to diff verdicts, and byte
// equivalence of the emitted code stays a valid exit gate for the registration.
//===----------------------------------------------------------------------===//

#include "triton/Analysis/TileAnalysis.h"
#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-tile-decide"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUTILEDECIDE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUTileDecidePass
    : public impl::TritonXPUTileDecideBase<TritonXPUTileDecidePass> {

public:
  using impl::TritonXPUTileDecideBase<
      TritonXPUTileDecidePass>::TritonXPUTileDecideBase;

  TritonXPUTileDecidePass() = default;
  TritonXPUTileDecidePass(bool reduceVec) { this->reduceVec = reduceVec; }

  // One partition per function, built before any root is decided: a per-value
  // state only means anything once the whole partition exists. The oracle is
  // the pre-tiling one -- every footprint question becomes a candidate instead
  // of a veto, which is exactly what makes the answer here E-independent.
  void buildVectorFlow(ModuleOp mod) {
    mod.walk([&](triton::FuncOp funcOp) {
      auto analysis = std::make_shared<VectorFlowAnalysis>(vectorFitUnknown);
      analysis->run(funcOp);
      vflowByFunc[funcOp.getOperation()] = analysis.get();
      vflowOwned.push_back(std::move(analysis));
    });
  }

  VectorFlowAnalysis *vflowFor(Operation *op) {
    auto funcOp = op->getParentOfType<triton::FuncOp>();
    if (!funcOp)
      return nullptr;
    auto it = vflowByFunc.find(funcOp.getOperation());
    return it == vflowByFunc.end() ? nullptr : it->second;
  }

  // The E-independent closure, i.e. the set the all-Unknown walk reaches. Used
  // only as `buildVecSetDomain`'s starting cone; the verdict below comes from
  // the partition, not from this walk.
  int64_t stateOnlyClosure(Operation *root, Type rootOpTy,
                           OperationTree &closure) {
    Type elemTy = getElementTypeOrSelf(getElementTypeOrSelf(rootOpTy));
    if (!isa<RankedTensorType>(rootOpTy) || !vectorizedTyValid(elemTy))
      return 0;
    // The reduce-operand gate is E-dependent and has no FitOracle signature, so
    // it is deferred the same way the footprint questions are: never veto here.
    auto reduceFitsPending = [](triton::xpu::ReduceOp, Type) { return true; };
    VectorizabilityAnalysis analysis(this->reduceVec, /*dumpFlag=*/false,
                                     reduceFitsPending, vectorFitUnknown);
    OperationTree visited;
    if (!analysis.getVectorizableClosure(root, visited, closure))
      closure.clear();
    return closure.size();
  }

  void decide(const char *site, Operation *root, Type rootOpTy,
              Value siteValue = {}) {
    if (!root)
      return;
    VectorFlowAnalysis *vflow = vflowFor(root);
    if (!vflow)
      return;

    // The value whose representation the verdict is about. Supplied by the
    // caller where it already holds it: a reduce operand can come from a
    // multi-result op, and guessing "the root's single result" drops those
    // roots.
    Value keyValue = siteValue;
    if (!keyValue) {
      if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(root))
        keyValue = storeOp.getValue();
      else if (root->getNumResults() == 1)
        keyValue = root->getResult(0);
    }
    if (!keyValue)
      return;

    OperationTree closure;
    int64_t closureSize = stateOnlyClosure(root, rootOpTy, closure);

    // Shared with the `[VecSet]` measurement on purpose: the decision has to be
    // taken over exactly the set that report prints, or the two drift silently.
    VecSetDomain domain = buildVecSetDomain(keyValue, *vflow, closure);

    // Tier 3's input. Packs and unpacks are counted separately and one by one
    // -- an op that ends up Vector fed by a Scalar class needs a pack on the
    // way in, a Scalar op fed by a Vector class needs an unpack -- because they
    // are priced separately and a segment need not hold whole pairs.
    int64_t matIn = 0, matOut = 0;
    for (Operation *op : domain.cone) {
      bool inTerm = domain.term.count(op);
      for (Value operand : op->getOperands()) {
        VState state = vflow->stateOf(operand);
        matIn += inTerm && state == VState::Scalar;
        matOut += !inTerm && state == VState::Vector;
      }
    }

    VState state = vflow->stateOf(keyValue);
    tileDecisionRecord(getOperation(), root, site, toString(state), closureSize,
                       domain.term.size(), matIn, matOut);

    StringRef kernel = "<unknown>";
    if (auto funcOp = root->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    // `tiers=` names what was and was not decided here. Printed on every line
    // rather than in a footnote: a verdict that only covers tier 3 must not be
    // read as the whole decision.
    llvm::errs() << "[TileDecide] " << kernel << " site=" << site
                 << " root=" << root->getName() << " state=" << toString(state)
                 << " tracked=" << vflow->isTracked(keyValue)
                 << " closure=" << closureSize << " cone=" << domain.cone.size()
                 << " term=" << domain.term.size() << " mat{in=" << matIn
                 << ",out=" << matOut << "}"
                 << " tiers={1:deferred,2:deferred,3:decided}"
                 << " loc=" << root->getLoc() << "\n";
  }

  void runOnOperation() override {
    if (!tileDecideEnabled())
      return;

    ModuleOp mod = getOperation();
    buildVectorFlow(mod);

    // Same enumeration as `tritonxpu-vectorizability-analysis`, in the same
    // order, so entry ids line up root-for-root with what that pass reports.
    if (this->reduceVec) {
      llvm::SetVector<triton::xpu::ReduceOp> reduceOps;
      mod.walk([&](triton::xpu::ReduceOp redOp) { reduceOps.insert(redOp); });
      for (auto redOp : reduceOps) {
        // An unvectorizable combine region is the reason a whole producer chain
        // stays scalar, so it is recorded rather than skipped -- otherwise the
        // decision's coverage is silently narrower than the walk's.
        if (!reduceCombineIsVectorizable(redOp)) {
          if (!redOp.getOperands().empty())
            decide("reduce-combine-veto", redOp, redOp.getInputTypes()[0],
                   redOp.getOperands()[0]);
          continue;
        }
        for (int i = 0; i < redOp.getOperands().size() - 1; ++i) {
          Value operand = redOp.getOperands()[i];
          decide("reduce-operand", operand.getDefiningOp(), operand.getType(),
                 operand);
        }
      }
    }

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      decide("store", storeOp, storeOp.getValue().getType());
    });
  }

private:
  // Owned per pass run; `vflowByFunc` only borrows. shared_ptr rather than
  // unique_ptr because MLIR's `clonePass()` copies the pass instance.
  llvm::SmallVector<std::shared_ptr<VectorFlowAnalysis>> vflowOwned;
  llvm::DenseMap<Operation *, VectorFlowAnalysis *> vflowByFunc;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
