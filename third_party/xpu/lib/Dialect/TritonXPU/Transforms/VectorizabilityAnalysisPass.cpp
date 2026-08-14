//===----------------------------------------------------------------------===//
// tritonxpu-vectorizability-analysis -- report-only pass over
// VectorizabilityAnalysis.
//
// Same purpose as TileAnalysisPass: give the measurement its own pipeline slot
// so the transform can later be handed the answer instead of recomputing it
// (redesign-v2.md §4.2 step 1.3). Never mutates the IR.
//
// It enumerates the roots Vectorize enumerates -- reduce operands under
// ReduceVec, then every store -- and reports, per root, whether the root is
// eligible and how big the closure is.
//
// Two instances of it are registered, and the point of the second one is to make
// a specific risk measurable instead of argued about.
//
//   preTiling=false -- immediately before Vectorize. Since step 1.5 moved that
//     pass' prologue into tritonxpu-normalize, this sees exactly the IR
//     Vectorize rewrites, so `stage=pre-vectorize` is by construction the answer
//     Vectorize will reach.
//
//   preTiling=true -- ahead of CoreTiling. The three in-walk footprint tests
//     (LoadOp, SplatOp, BroadcastOp) read sizePerCore, which does not exist yet,
//     so this instance hands the walk an all-Unknown oracle: it never vetoes and
//     records what it deferred (`stage=pre-tiling`, `cands=`). It then repeats
//     the walk with the real oracle at the same position (`stage=pre-tiling-fit`).
//
// `pre-tiling-fit` vs `pre-vectorize` is the diff that matters: it holds the walk
// fixed and varies only the position, so a disagreement is CoreTiling or Legalize
// moving an E-dependent answer -- e.g. Legalize.cpp:245-247's
// `slicedShape[i] = max(shape[i]/iterCount[i], 1)`, whose saturation can flip
// BroadcastOp's `srcShape[1] == resShape[1]` from false to true. Agreement is
// what licenses moving the state half up (step 1.5c).
//
// Nothing is emitted unless TRITONXPU_VEC_REPORT=1, and neither instance mutates
// the IR.
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Analysis/TileAnalysis.h"
#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-vectorizability-analysis"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUVECTORIZABILITYANALYSIS
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUVectorizabilityAnalysisPass
    : public impl::TritonXPUVectorizabilityAnalysisBase<
          TritonXPUVectorizabilityAnalysisPass> {

public:
  using impl::TritonXPUVectorizabilityAnalysisBase<
      TritonXPUVectorizabilityAnalysisPass>::
      TritonXPUVectorizabilityAnalysisBase;

  TritonXPUVectorizabilityAnalysisPass() = default;
  TritonXPUVectorizabilityAnalysisPass(bool reduceVec, bool preTiling) {
    this->reduceVec = reduceVec;
    this->preTiling = preTiling;
  }

  // Today's answer, computed with the real oracle at whatever position this
  // instance sits. A fresh analysis per root, matching
  // vectorizeAndProcessOpVecTy: sharing `visited` across roots changes the
  // answer, and this pass has to report what that function would see, not
  // something tidier.
  std::pair<bool, int64_t> reportWithFit(const char *stage, const char *site,
                                        Operation *root, Type rootOpTy) {
    VectorizabilityAnalysis analysis(this->reduceVec, /*dumpFlag=*/false,
                                     vectorFitsReduceOperand, vectorFitsValue);
    bool eligible = vectorFitsRoot(rootOpTy);
    int64_t closureSize = 0;
    if (eligible) {
      OperationTree visited, vectorizedOps;
      if (analysis.getVectorizableClosure(root, visited, vectorizedOps))
        closureSize = vectorizedOps.size();
    }
    if (vecReportEnabled())
      reportVecRoot(stage, site, root, rootOpTy, eligible, closureSize);
    return {eligible, closureSize};
  }

  // What a state-only walk can conclude before CoreTiling: the all-Unknown
  // oracle never vetoes, so `closure` here is the E-independent answer and
  // `cands` is how many footprint questions were deferred.
  void reportState(const char *site, Operation *root, Type rootOpTy) {
    // Eligibility keeps only `vectorFitsRoot`'s E-independent conjunct. The
    // other two (numElems >= width, numElems % width == 0) divide the per-core
    // element count, so they belong to the deferred side -- but unlike the three
    // in-walk cases they have no oracle to go through yet, which is why the root
    // itself is not in `cands`. That is the hole step 2.1 has to close.
    Type elemTy = getElementTypeOrSelf(getElementTypeOrSelf(rootOpTy));
    bool eligible =
        isa<RankedTensorType>(rootOpTy) && vectorizedTyValid(elemTy);

    // The reduce-operand gate is E-dependent too and its signature is
    // (ReduceOp, Type) rather than a value footprint, so it cannot be a
    // FitCandidate. Not vetoing here and counting it into `cands` keeps the
    // "deferred, not decided" accounting honest.
    int64_t redPending = 0;
    auto reduceFitsPending = [&](triton::xpu::ReduceOp, Type) {
      ++redPending;
      return true;
    };

    VectorizabilityAnalysis analysis(this->reduceVec, /*dumpFlag=*/false,
                                     reduceFitsPending, vectorFitUnknown);
    int64_t closureSize = 0;
    int64_t cands = 0;
    if (eligible) {
      OperationTree visited, vectorizedOps;
      if (analysis.getVectorizableClosure(root, visited, vectorizedOps)) {
        closureSize = vectorizedOps.size();
        cands = analysis.getFitCandidates().size() + redPending;
      }
    }
    if (vecReportEnabled())
      reportVecRoot("pre-tiling", site, root, rootOpTy, eligible, closureSize,
                    cands);
  }

  // Step 2.1: the same roots, answered by the Vector-Flow partition instead of
  // by the closure walk. One analysis per function, built before any root is
  // reported, because a per-value answer is only meaningful once the whole
  // partition exists.
  VectorFlowAnalysis *vflowFor(Operation *op) {
    auto funcOp = op->getParentOfType<triton::FuncOp>();
    if (!funcOp)
      return nullptr;
    auto it = vflowByFunc.find(funcOp.getOperation());
    return it == vflowByFunc.end() ? nullptr : it->second;
  }

  void buildVectorFlow(ModuleOp mod) {
    mod.walk([&](triton::FuncOp funcOp) {
      auto analysis = std::make_shared<VectorFlowAnalysis>(vectorFitsValue);
      analysis->run(funcOp);
      const VectorFlowStats &st = analysis->getStats();
      llvm::errs() << "[VFlow] " << funcOp.getName()
                   << " summary values=" << st.values
                   << " classes=" << st.classes << " vector=" << st.vectorClasses
                   << " scalar=" << st.scalarClasses
                   << " conflict=" << st.conflictClasses
                   << " unset=" << st.unsetClasses << " unions=" << st.unions
                   << " pins{vec=" << st.vectorPins << ",scl=" << st.scalarPins
                   << ",extern=" << st.externPins
                   << ",unknown=" << st.unknownPins << "}"
                   << " reduce{ops=" << st.reduceOps
                   << ",vecEntry=" << st.reduceVectorEntries
                   << ",entryUnpack=" << st.reduceEntryUnpacks << "}\n";
      vflowByFunc[funcOp.getOperation()] = analysis.get();
      vflowOwned.push_back(std::move(analysis));
    });
  }

  // `keyValue` is the value whose state the closure verdict is about: the
  // store's value operand, or the reduce operand. Agreement is the contract this
  // step has to report -- a disagreement is either the partition seeing a
  // boundary the walk had to veto on (expected, that is the point) or a bug.
  void reportVFlowRoot(const char *site, Operation *root, Value keyValue,
                       int64_t closureSize) {
    VectorFlowAnalysis *analysis = vflowFor(root);
    if (!analysis || !keyValue)
      return;
    StringRef kernel = "<unknown>";
    if (auto funcOp = root->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    VState state = analysis->stateOf(keyValue);
    bool walkSaysVector = closureSize > 0;
    bool flowSaysVector = state == VState::Vector;
    // Step 2.2 makes one class of disagreement expected, so it is named rather
    // than counted as agreement. The walk is all-or-nothing and vetoes at a
    // reduce whose combine region cannot be retyped; the partition answers
    // Vector for the producer chain and puts a boundary at the reduce entry.
    // That is the modelled case (`boundary`). Anything else that disagrees --
    // in particular the walk retyping a chain the partition calls Scalar -- is
    // a real `MISMATCH`, greppable in upper case on purpose.
    bool tracked = analysis->isTracked(keyValue);
    const char *kind = "match";
    if (walkSaysVector != flowSaysVector) {
      if (!walkSaysVector && flowSaysVector &&
          StringRef(site) == "reduce-combine-veto")
        kind = "boundary";
      else if (walkSaysVector && state == VState::Unset)
        // Unpinned, not contradicted: no seed in this class demands either
        // representation. welford's `w` accumulator is the case -- it is built
        // from constants and loop-carried values only, so the sole reason the
        // walk retypes it is the reduce being a vector consumer, and *that* is
        // the E-dependent fit question (`vectorFitsReduceOperand`) this analysis
        // deliberately does not answer. Reported as its own kind rather than
        // folded into agreement: a free class is a candidate, which is all M1
        // promises, but it is not the same statement as "Vector".
        kind = tracked ? "free" : "UNTRACKED";
      else
        kind = "MISMATCH";
    }
    llvm::errs() << "[VFlow] " << kernel << " root site=" << site
                 << " root=" << root->getName() << " closure=" << closureSize
                 << " state=" << toString(state) << " tracked=" << tracked
                 << " agree=" << (walkSaysVector == flowSaysVector)
                 << " kind=" << kind << " loc=" << root->getLoc() << "\n";
  }

  void report(const char *site, Operation *root, Type rootOpTy,
              Value siteValue = {}) {
    if (!root)
      return;
    if (!this->preTiling) {
      auto [eligible, closureSize] =
          reportWithFit("pre-vectorize", site, root, rootOpTy);
      if (vflowReportEnabled()) {
        // The value whose representation the verdict is about. `siteValue` is
        // supplied where the caller already holds it -- a reduce operand can be
        // defined by a multi-result op (layernorm's rebuilt `scf.for`), and
        // guessing "the root's single result" silently drops those roots.
        Value keyValue = siteValue;
        if (!keyValue) {
          if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(root))
            keyValue = storeOp.getValue();
          else if (root->getNumResults() == 1)
            keyValue = root->getResult(0);
        }
        reportVFlowRoot(site, root, keyValue, closureSize);
      }
      // This instance sits where M1/M2 will produce, so it is the one that puts
      // the conclusion into the plan (step 1.6). The pre-tiling instance must
      // not: it would key roots that the walk there cannot even see, and its ids
      // would collide with these.
      tilePlanRecord(getOperation(), root, site, eligible, closureSize);
      return;
    }
    reportState(site, root, rootOpTy);
    // Same walk, real oracle, same position: today's answer as it would come
    // out *here*. Diffing this against stage=pre-vectorize isolates one thing
    // and nothing else -- whether CoreTiling and Legalize move the E-dependent
    // answer between the two positions. If they do not, the state side above is
    // free to move up; if they do, `cands` is where the difference has to be
    // re-tested rather than assumed.
    reportWithFit("pre-tiling-fit", site, root, rootOpTy);
  }

  void runOnOperation() override {
    // Three independent switches: the report, the plan probe, and the step 2.1
    // partition. Any one of them being on is reason enough to walk.
    if (!vecReportEnabled() && !(tilePlanProbeEnabled() && !this->preTiling) &&
        !(vflowReportEnabled() && !this->preTiling))
      return;

    ModuleOp mod = getOperation();

    // Before any root is reported: the per-root lines below read this.
    if (vflowReportEnabled() && !this->preTiling)
      buildVectorFlow(mod);

    if (this->reduceVec) {
      llvm::SetVector<triton::xpu::ReduceOp> reduceOps;
      mod.walk([&](triton::xpu::ReduceOp redOp) { reduceOps.insert(redOp); });
      for (auto redOp : reduceOps) {
        // Reported rather than skipped: an unvectorizable combine region is the
        // reason a whole producer chain stays scalar, so it is the interesting
        // case, and Vectorize's own `continue` here is measured to be redundant
        // (the closure walk vetoes at the reduce anyway).
        if (!reduceCombineIsVectorizable(redOp)) {
          if (vecReportEnabled())
            reportVecRoot(this->preTiling ? "pre-tiling" : "pre-vectorize",
                          "reduce-combine-veto", redOp,
                          redOp.getInputTypes()[0], /*eligible=*/false,
                          /*closureSize=*/0);
          if (!this->preTiling) {
            tilePlanRecord(mod, redOp, "reduce-combine-veto",
                           /*eligible=*/false, /*closure=*/0);
            // Vetoed roots are reported too, otherwise the partition's coverage
            // is silently narrower than the walk's.
            if (vflowReportEnabled() && !redOp.getOperands().empty())
              reportVFlowRoot("reduce-combine-veto", redOp,
                              redOp.getOperands()[0], /*closureSize=*/0);
          }
          continue;
        }
        for (int i = 0; i < redOp.getOperands().size() - 1; ++i) {
          Value operand = redOp.getOperands()[i];
          report("reduce-operand", operand.getDefiningOp(), operand.getType(),
                 operand);
        }
      }
    }

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      report("store", storeOp, storeOp.getValue().getType());
    });
  }

private:
  // Owned per pass run; `vflowByFunc` only borrows. Cleared implicitly when the
  // pass instance dies, which is why nothing here outlives the report.
  // shared_ptr rather than unique_ptr because MLIR's `clonePass()` copies the
  // pass instance, and a unique_ptr member would make the pass non-copyable.
  llvm::SmallVector<std::shared_ptr<VectorFlowAnalysis>> vflowOwned;
  llvm::DenseMap<Operation *, VectorFlowAnalysis *> vflowByFunc;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
