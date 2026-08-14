#ifndef TRITONXPU_ANALYSIS_VECTORIZABILITYANALYSIS_H
#define TRITONXPU_ANALYSIS_VECTORIZABILITYANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Analysis/NewAnalysis/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#include <functional>

//===----------------------------------------------------------------------===//
// Vectorizability analysis: which ops can take a `vector<N x T>` element type,
// and how wide that vector is.
//
// This answers the question the Vectorize pass used to answer inline while it
// was already rewriting types. Separating the two matters because the tile
// factorization
//
//     E = vectorWidth * vregsPerIter * iterNum
//
// couples Vectorize to UnrollControl: whether a tree vectorizes decides whether
// its values sit in 512-bit registers or in scalar ones, which decides the
// register pressure UnrollControl has to tile against.
//
// The predicates split into two groups, and step 1.4 of the redesign has made
// that split structural: the E-dependent half now lives in TileAnalysis.h
// (`vectorFitsRoot`, `vectorFitsReduceOperand`) and this unit only reaches it
// through the callback handed to the constructor. Nothing here includes
// TileAnalysis.h, which is what lets step 1.5 move this unit ahead of CoreTiling.
//
//   * E-independent, so answerable before CoreTiling fixes sizePerCore: the
//     op-kind whitelist in `getVectorizableClosure`, the element type whitelist
//     in `vectorizedTyValid`, closure connectivity, the reduce combine-region
//     check, and the ExternElementwise symbol whitelist.
//   * E-dependent, so only answerable once sizePerCore is known: the root and
//     reduce-operand fit tests, both of which divide an element count by the
//     vector width. Now in TileAnalysis.
//
// Step 1.5b finished the split for the three cases whose residue sat inside the
// walk rather than at its edges: LoadOp's numElems % vectorWidth == 0, SplatOp's
// getTotalElemsPerThread(srcTy) == 1, and BroadcastOp's result element count all
// go through the `FitOracle` below now, so the walk itself no longer calls
// getTotalElemsPerThread. What each case keeps is its E-independent half (op
// kind, rank, shape relations, element width). Moving the walk ahead of
// CoreTiling is then a matter of handing it an all-`Unknown` oracle and
// re-testing the candidates in place, which is step 1.5c.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace xpu {

using OperationTree = llvm::SetVector<mlir::Operation *>;

#define ARITH_BINARY_FLOAT_OP                                                  \
  arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,                  \
      arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp

#define ARITH_BINARY_INT_OP                                                    \
  arith::SubIOp, arith::AndIOp, arith::OrIOp, arith::MulIOp, arith::AddIOp,    \
      arith::XOrIOp

#define MATH_UNARY_OP                                                          \
  math::ExpOp, math::SqrtOp, math::SinOp, math::CosOp, arith::ExtFOp,          \
      arith::TruncFOp, math::AbsFOp, math::LogOp

#define REDUCE_COMBINE_OP COMBINE_OP, triton::xpu::ReduceReturnOp

// The scalar -> vector op table, single-sourced (redesign-v2.md §3.1, step 2.1).
//
// Vectorize.cpp expands this into the `VOp<T>` specializations its rewrite
// builds from; `hasVectorForm` below expands the same list into a runtime
// predicate. Keeping the second consumer as a hand-maintained whitelist is what
// §3.1 forbids, and the cost of drift is not a missed optimization: the rewrite
// dispatch ends in `llvm_unreachable`, so a table the analysis believes in but
// the rewrite lacks is a crash.
//
// Membership here means only "this op kind has a 512-bit vector form". Element
// type (`vectorizedTyValid`), footprint (the FitOracle) and operand shape are
// separate questions asked elsewhere.
#define TTX_SCALAR_TO_VECTOR_OPS(FN)                                           \
  FN(arith::AddFOp, triton::xpu::VvaddFOp)                                     \
  FN(arith::SubFOp, triton::xpu::VvsubFOp)                                     \
  FN(arith::MulFOp, triton::xpu::VvmulFOp)                                     \
  FN(arith::DivFOp, triton::xpu::VvdivFOp)                                     \
  FN(arith::MaximumFOp, triton::xpu::VvmaxFOp)                                 \
  FN(arith::MinimumFOp, triton::xpu::VvminFOp)                                 \
  FN(arith::MaxNumFOp, triton::xpu::VvmaxNumFOp)                               \
  FN(arith::MinNumFOp, triton::xpu::VvminNumFOp)                               \
  FN(arith::AddIOp, triton::xpu::VvaddIOp)                                     \
  FN(arith::SubIOp, triton::xpu::VvsubIOp)                                     \
  FN(arith::MulIOp, triton::xpu::VvmulIOp)                                     \
  FN(arith::AndIOp, triton::xpu::VvandIOp)                                     \
  FN(arith::XOrIOp, triton::xpu::VvxorIOp)                                     \
  FN(arith::OrIOp, triton::xpu::VvorIOp)                                       \
  FN(math::ExpOp, triton::xpu::VExpFOp)                                        \
  FN(math::AbsFOp, triton::xpu::VAbsFOp)                                       \
  FN(math::LogOp, triton::xpu::VLogFOp)                                        \
  FN(math::SqrtOp, triton::xpu::VSqrtFOp)                                      \
  FN(math::SinOp, triton::xpu::VSinFOp)                                        \
  FN(math::CosOp, triton::xpu::VCosFOp)                                        \
  FN(arith::ExtFOp, triton::xpu::VExtFOp)                                      \
  FN(arith::TruncFOp, triton::xpu::VTruncFOp)                                  \
  FN(arith::SIToFPOp, triton::xpu::VSIToFPOp)

// Does this op kind have a vector form, per the table above? Op kind only.
bool hasVectorForm(Operation *op);

// Is the region-interpreting reduce lowering in use?
//
// **Default off.** It was flipped on 2026-08-05 and flipped back on 2026-08-06:
// with it on, a *vectorized* welford reduce silently drops the lane-0
// contribution of most cores, and which cores are affected changes from run to
// run. Minimal repro (FlagGems `var_mean_kernel_2`, one f32 vector per core):
// feed every partial `acc=1, average=0, count=1` so the exact fold must yield
// `nvar == BLOCK_NUM`; at BLOCK_NUM=1024 it yields 970 / 977 on successive runs,
// and sweeping the payload lane by lane shows the losses land exactly on the
// multiples of 16, i.e. the seed element `collapseVectorsJointly` extracts
// first. BLOCK_NUM <= 128 puts fewer than 16 elements on a core, skips the
// within-core fold, and stays exact. Observable as
// `test_accuracy_varmean[dtype1-*-*-{dim2,dim3}-shape1]` in
// third_party/xpu/test/FlagGems/tests/test_reduction_ops.py: those eight are the
// only regressions in that 3187-case suite, and TRITONXPU_REDUCE_REGION=0 fixes
// all eight while TRITONXPU_BUDGET_TILING=0 fixes none.
//
// The default is off rather than the admission gate in
// ReduceOpToLLVM::canInterpretCombine being narrowed, because the run-to-run
// variation means the real boundary is not known yet; a gate drawn around the
// one shape we happened to measure would be a made-up bound.
//
// `TRITONXPU_REDUCE_REGION=1` opts back in. The measured upside is real and
// waiting on that defect, all on xpu3 hardware: welford 801.4us vs 1319us all
// scalar = 1.65x (findings.md 1.17), pairmax 910.6us vs 1058.8us = 1.16x (1.22),
// and of the eleven golden probes only those two plus `bitred` change code at all
// -- the other eight are byte-identical either way, and `bitred` only differs in
// emission order (same instruction multiset). Combine-op coverage is closed in
// 1.24: twelve of the thirteen ops `isSupportedCombineOp` admits have a probe.
//
// Single-sourced here because three units have to agree on it: this analysis (op
// whitelist), Vectorize (region retyping), and ReduceOpToLLVM (which lowering to
// emit). A disagreement does not degrade, it hits llvm_unreachable.
inline bool reduceCombineRegionEnabled() {
  static const bool enabled =
      mlir::triton::tools::isEnvValueBool(
          mlir::triton::tools::getStrEnv("TRITONXPU_REDUCE_REGION"))
          .value_or(false);
  return enabled;
}

// The ops the region-interpreting lowering can also emit
// (ReduceOpToLLVM::emitCombineOp). Only reachable while the region lowering is
// on: with TRITONXPU_REDUCE_REGION=0 the lowering applies one op per output and
// these would be silently dropped. Constants are deliberately left scalar by the
// retyping and splatted at the use site.
//
// arith::NegFOp is deliberately absent: Triton's unary minus lowers to
// `subf(0.0, x)`, so no combine region can contain a NegFOp (findings.md 1.24).
#define REDUCE_COMBINE_REGION_OP                                               \
  REDUCE_COMBINE_OP, arith::SubFOp, arith::DivFOp, arith::SelectOp,            \
      arith::ConstantOp

// Element types that have a 512-bit vector form on this target. Note i1 is
// absent, which is why a bool store never vectorizes.
bool vectorizedTyValid(Type elemTy);

// Lanes a 512-bit vector register holds for `elemTy`. There is no freedom here
// today: the width is fully determined by the element width.
unsigned getVectorWidth(Type elemTy);

// Whether every op in the reduce's combine region has a vector form, i.e.
// whether `Vectorize`'s wholesale retyping of that region can succeed.
//
// Single-sourced on purpose: the analysis and the rewrite both have to agree on
// REDUCE_COMBINE_OP, and when they disagree the rewrite does not fall back, it
// hits llvm_unreachable.
//
// E-independent: only looks at op kinds.
bool reduceCombineIsVectorizable(triton::xpu::ReduceOp redOp);

// Root-level report, shared by `tritonxpu-vectorizability-analysis` and by
// Vectorize's own root enumeration.
//
// It exists because the two see different IR and the difference has to be
// diffable rather than assumed: the standalone pass runs before Vectorize, so
// before that pass' prologue (erf lowering, maximum/compare fusion, i1 logic to
// i8), and those rewrites change which roots form a closure. Emitting the same
// line format from both sides is what makes the gap step 1.5 has to close
// visible. Off unless TRITONXPU_VEC_REPORT=1.
bool vecReportEnabled();
// `cands` is the number of footprint questions the walk left open (step 1.5c);
// -1 means "not applicable", which is every caller running the real oracle.
void reportVecRoot(const char *stage, const char *site, Operation *root,
                   Type rootOpTy, bool eligible, int64_t closureSize,
                   int64_t cands = -1);

//===----------------------------------------------------------------------===//
// The fit oracle (redesign-v2.md §2.1.1, step 1.5b).
//
// Three cases inside the closure walk -- LoadOp, SplatOp, BroadcastOp -- ask
// about the per-core footprint of a value, which is exactly what CoreTiling
// fixes. Rather than invent a pre-tiling approximation for them, each case now
// keeps only its E-independent half and asks an injected oracle for the rest.
//
// `Unknown` is the pre-tiling answer: the walk must not veto on it, and records
// the query as a `FitCandidate` instead (step 1.5c) so the E-dependent half can
// be applied once sizePerCore exists. On the E-dependent side the real oracle
// answers Yes/No only, so the walk is byte-identical to the inline tests it
// replaces and records nothing.
//
// The width is not the oracle's business -- it follows from the element type
// alone, so the walk computes it and passes it in. That deliberately keeps
// inconsistency #3 of §2.1.1 (BroadcastOp hardcodes 16, right only for f32)
// visible at the call site instead of burying it in the oracle.
//===----------------------------------------------------------------------===//

enum class Fit { Yes, No, Unknown };

// What is being asked of the value's per-core footprint:
//   WholeVectors -- a nonzero whole number of `wantWidth`-lane vectors
//   SingleElem   -- exactly one element (`wantWidth` unused)
//   AtLeastWidth -- at least `wantWidth` elements
enum class FitQuery { WholeVectors, SingleElem, AtLeastWidth };

using FitOracle = std::function<Fit(Value, FitQuery, unsigned)>;

// One footprint question the walk could not answer, because the oracle it was
// handed returned `Unknown`. The walk does not veto on these -- it records them
// and keeps going, so the closure it reports is the E-independent answer and
// this list is exactly what still has to be checked against E.
//
// All-or-nothing on purpose, matching today's behavior: any candidate that
// later fails the real fit vetoes the whole tree, because that is what the
// inline test it replaces did.
struct FitCandidate {
  Value value;
  FitQuery query;
  unsigned wantWidth;
};

// The pre-tiling oracle: knows nothing, so every query becomes a candidate.
Fit vectorFitUnknown(Value value, FitQuery query, unsigned wantWidth);

class VectorizabilityAnalysis {
public:
  // `reduceOperandFits` is the E-dependent gate the ReduceOp case needs, passed
  // in rather than called directly so this unit keeps no compile-time dependency
  // on TileAnalysis. Callers pass `vectorFitsReduceOperand`; it is owned, not
  // borrowed, because every caller builds the analysis from a temporary.
  VectorizabilityAnalysis(
      bool reduceVec, bool dumpFlag,
      std::function<bool(triton::xpu::ReduceOp, Type)> reduceOperandFits,
      FitOracle fitOracle)
      : ReduceVec(reduceVec), dumpFlag(dumpFlag),
        reduceOperandFits(std::move(reduceOperandFits)),
        fitOracle(std::move(fitOracle)) {}

  // Every op that can be retyped if `root` is, or an empty result when the
  // closure cannot be formed. The closure is bidirectional (operands and users)
  // and any single unsupported op vetoes the whole tree.
  bool getVectorizableClosure(Operation *root, OperationTree &visited,
                              OperationTree &vectorizedOps) {
    fitCandidates.clear();
    return vectorize(root, visited, vectorizedOps);
  }

  // The footprint questions this walk left open, valid until the next
  // `getVectorizableClosure` call. Empty with the real oracle. Only meaningful
  // when the walk succeeded: a vetoed branch can leave candidates behind, and
  // since a veto propagates to the root, a failed closure discards them.
  const llvm::SmallVector<FitCandidate> &getFitCandidates() const {
    return fitCandidates;
  }

private:
  // Ask the oracle, recording rather than vetoing on `Unknown`.
  Fit askFit(Value value, FitQuery query, unsigned wantWidth);

  Operation *getBlockArgumentOp(Value arg);
  bool binLikeOpVectorize(Value lhs, Value rhs, OperationTree &visited,
                          OperationTree &vectorizedOps);
  bool vectorize(Operation *op, OperationTree &visited,
                 OperationTree &vectorizedOps);

  bool ReduceVec;
  bool dumpFlag;
  std::function<bool(triton::xpu::ReduceOp, Type)> reduceOperandFits;
  FitOracle fitOracle;
  llvm::SmallVector<FitCandidate> fitCandidates;
};

//===----------------------------------------------------------------------===//
// Vector-Flow analysis (M1, redesign-v2.md §3.1, step 2.1).
//
// The closure walk above answers one question per root: "can this whole tree be
// retyped, yes or no". Anything it cannot retype vetoes the entire tree, which
// is why a single `arith.cmpi` in a store chain costs the chain its vector form.
//
// This unit answers a different question, per SSA value rather than per root:
// which state does this value *want*. Values that must agree are unioned into
// one class; a class holding both a Vector and a Scalar pin is not an error, it
// is a boundary -- somewhere on that class' edge a pack/unpack has to go. M2
// decides where; this unit only reports.
//
// Termination is structural, not a fixed point: the propagation is union-find
// over equality edges, so every step strictly reduces the class count and there
// is nothing to iterate. That is a deliberate deviation from the "sparse
// DataFlowAnalysis" base §3.1 names -- the constraints it lists are all
// symmetric ("operands and result same state", "init <-> iter_arg <-> yield <->
// result"), and a symmetric constraint system is a partition, not a lattice
// climb. It also sidesteps §4.2's convergence worry on control flow outright.
//
// Nothing consumes the result yet, so the exit gate is C1 byte equality.
//===----------------------------------------------------------------------===//

// Unset <= {Vector, Scalar} <= Conflict, as in §3.1.
enum class VState { Unset, Vector, Scalar, Conflict };

const char *toString(VState state);

struct VectorFlowStats {
  int64_t values = 0;          // SSA values reached
  int64_t classes = 0;         // equality classes after unioning
  int64_t vectorClasses = 0;   // ... of which pinned Vector only
  int64_t scalarClasses = 0;   // ... Scalar only
  int64_t conflictClasses = 0; // ... both, i.e. a boundary
  int64_t unsetClasses = 0;    // ... neither, free to go either way
  int64_t unions = 0;          // equality edges that actually merged
  int64_t vectorPins = 0;
  int64_t scalarPins = 0;
  int64_t externPins = 0;  // extern_elementwise, pinned Scalar for now
  int64_t unknownPins = 0; // op kind not modelled, pinned Scalar
  // Step 2.2: the reduce entry as a boundary rather than a veto.
  int64_t reduceOps = 0;           // reduces with at least one data operand
  int64_t reduceVectorEntries = 0; // ... data operands whose class came out Vector
  int64_t reduceEntryUnpacks = 0;  // ... of those, the ones needing a real unpack
};

class VectorFlowAnalysis {
public:
  // The oracle is the same one the closure walk takes, and for the same reason:
  // whether a load's footprint is a whole number of vectors is E-dependent, so
  // it must not be answered here. `Unknown` seeds nothing.
  explicit VectorFlowAnalysis(FitOracle fitOracle)
      : fitOracle(std::move(fitOracle)) {}

  // Partition every value in `func`, then pin. Read-only on the IR.
  void run(triton::FuncOp func);

  // Unset for values `run` never saw.
  VState stateOf(Value value) const;

  // Whether `run` reached this value at all. `stateOf` folds two different
  // answers into Unset -- "reached, and nothing constrains it" versus "never
  // reached" -- and only the first is a usable answer. The per-root report
  // prints both so an unmodelled value cannot pass as an unconstrained one.
  bool isTracked(Value value) const { return ids.count(value) != 0; }

  const VectorFlowStats &getStats() const { return stats; }

private:
  unsigned idOf(Value value);
  unsigned find(unsigned id);
  void unite(Value a, Value b);
  void pin(Value value, VState state);
  void visit(Operation *op);

  FitOracle fitOracle;
  llvm::DenseMap<Value, unsigned> ids;
  llvm::SmallVector<unsigned> parent;
  llvm::SmallVector<VState> pins; // per class root, valid after `find`
  VectorFlowStats stats;
};

// Off unless TRITONXPU_VFLOW_REPORT=1.
bool vflowReportEnabled();

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITONXPU_ANALYSIS_VECTORIZABILITYANALYSIS_H
