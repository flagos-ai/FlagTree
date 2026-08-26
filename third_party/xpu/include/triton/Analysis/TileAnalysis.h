#ifndef TRITONXPU_ANALYSIS_TILEANALYSIS_H
#define TRITONXPU_ANALYSIS_TILEANALYSIS_H

#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

//===----------------------------------------------------------------------===//
// Tile analysis: the measurements a tiling decision needs, kept separate from
// the passes that act on them.
//
// A tile of an XPU kernel is described by three factors whose product is the
// per-core element count fixed by CoreTiling:
//
//     E = vectorWidth * vregsPerIter * iterNum
//
// `vectorWidth` is chosen by Vectorize, `iterNum` by UnrollControl, and
// `vregsPerIter` is the residual that decides whether the tile spills. The
// passes that pick those factors are transforms; what they need in order to
// pick is measurement, and that is what lives here so a single planner can
// eventually consume all of it.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace xpu {

// Cluster layout of a tensor type, looking through the slice encoding that a
// reduction leaves on its result.
ClusterLayoutAttr getClusterLayout(RankedTensorType tensorTy);

// Per-core register footprint of one SSA value, counted in registers of its own
// file. `isVector`, when given, reports which file that is.
//
// Vector values need no conversion: Vectorize divides sizePerCore.back() by the
// vector width, so product(sizePerCore) already is the number of vector
// registers held.
//
// Scalar-element values are counted too, one scalar register per element.
// Vectorize bails out whenever numElems < vectorWidth or
// numElems % vectorWidth != 0 (Vectorize.cpp:335), so a partially vectorized
// tree keeps scalar tensors live across the whole segment and used to be
// invisible here. The two files are kept apart on purpose: a vector register is
// 512 bits and a scalar one 32, they spill independently, and folding both into
// one budget makes the target meaningless (measured: it collapses layernorm to
// iterNum=1 and 23 vector spills).
int64_t getNumRegs(Type type, bool *isVector = nullptr);

struct RegPressure {
  int64_t vecPeak = 0;    // 512-bit registers simultaneously live
  int64_t scalarPeak = 0; // 32-bit registers simultaneously live
  // Whole footprint the segment defines, which is what the un-tiled full width
  // has to keep alive.
  int64_t vecTotal = 0;
  int64_t scalarTotal = 0;
  // Widest per-core last dim among *vector* values only. Widths of the two
  // files are in different units (registers vs elements) and must never be
  // maxed together, or a width conversion against this divides the target down
  // to 1.
  int64_t maxVecWidth = 1;
  // Narrowest per-core last dim among *vector* values, in vector slots, or 0
  // when the segment holds none. This is a legality bound, not a heuristic: the
  // retyping slices sizePerCore with ceil(), so a trip count larger than a
  // vector value's slot count saturates that value at one slot per iteration
  // while the scalar values in the same tree keep dividing. The two sides then
  // disagree on how many lanes one iteration covers, which is how a vselect
  // ends up reading its condition past the end
  // (VectorizedOpToLLVM.cpp:539-600 assumes condElems == numElems * vecSize).
  int64_t minVecWidth = 0;
};

// Peak simultaneously-live registers across `opTree` as it stands, per file.
// `m` supplies the program order the liveness walk needs.
void getRegPressure(ModuleOp m, const llvm::SetVector<Operation *> &opTree,
                    RegPressure &p);

// Same measurement over the whole block containing `insertPt`. Trees of
// different widths sharing a block must be tiled to the same per-iteration
// width rather than the same trip count, so the block-wide maxVecWidth matters
// as much as its peak.
void getBlockRegPressure(ModuleOp m, Operation *insertPt, RegPressure &p);

//===----------------------------------------------------------------------===//
// E-dependent gates.
//
// These two read sizePerCore, so they cannot be answered before CoreTiling has
// fixed E. They used to sit in VectorizabilityAnalysis next to the
// E-independent op-kind and element-type whitelists; they live here instead so
// the state half of the vectorizability question carries no dependency on the
// numeric half and can be moved ahead of CoreTiling on its own (redesign-v2.md
// §4.2 steps 1.4 and 1.5).
//
// Both are phrased as "does a whole number of 512-bit vectors fit", which is
// the only thing either of them tests. Neither had any state, which is what
// makes them free functions here.
//===----------------------------------------------------------------------===//

// Whether a root tensor type holds enough elements per core, in a whole number
// of vectors, to be worth vectorizing at all. `rowsPerCore` is factored out
// first, so a core holding several rows is judged on one row's width.
bool vectorFitsRoot(Type rootOpTy);

// Same question for one operand of a reduce, measured along the reduced axis.
// Note this reads the *tensor shape* on that axis, not sizePerCore, so it is
// E-dependent only through the layout CoreTiling leaves behind.
bool vectorFitsReduceOperand(triton::xpu::ReduceOp redOp, Type operandTy);

// The real fit oracle handed to `VectorizabilityAnalysis` (step 1.5b): answers
// the three footprint questions the closure walk used to answer inline by
// calling `getTotalElemsPerThread` itself. E-dependent, hence here; never
// returns `Fit::Unknown`, which is what makes the walk byte-identical to before
// on this side of CoreTiling.
Fit vectorFitsValue(Value value, FitQuery query, unsigned wantWidth);

//===----------------------------------------------------------------------===//
// M6 -- the tile plan carrier (redesign-v2.md §3.7, step 1.6).
//
// The analysis that decides vectorizability runs before `tritonxpu-vectorize`;
// the transforms that consume the decision run around
// `tritonxpu-unroll-control`, with vectorize / canonicalize / alloca /
// memory-async in between. Something has to survive that gap and still identify
// *which root* each conclusion belongs to.
//
// §3.7 named two candidate keys and marked both unverified. This is the probe
// that decides between them by measuring, not by argument: the producer writes
// both keys for every root, the consumer looks both up again and reports two
// hit rates.
//
//   loc     -- the root's location, printed. Free, but a pass that rebuilds an
//   op
//              has to carry the loc across for this to work.
//   plan_id -- an integer the producer stamps on the root op itself. Stable
//              against renaming, but only survives if the op survives *and* its
//              discardable attributes are copied.
//
// The payload is deliberately uninteresting (whatever the producing site
// already computed). Step 1.6 measures identification, not content.
//
// Everything here is off unless TRITONXPU_TILE_PLAN=1, and `tilePlanCheck`
// erases both keys unconditionally -- an entry reaching the emitted IR would
// break C1, which is the other half of this step's exit gate.
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kTilePlanAttrName = "triton_xpu.tile_plan";
constexpr llvm::StringLiteral kTilePlanIdAttrName = "triton_xpu.plan_id";

bool tilePlanProbeEnabled();

// Append one entry for `root`, stamping it with the next id. No-op when the
// probe is off.
void tilePlanRecord(ModuleOp mod, Operation *root, StringRef site,
                    bool eligible, int64_t closure);

// Resolve every entry against the IR as it now stands, report per-entry how
// many live ops each key found, and erase everything the probe wrote.
void tilePlanCheck(ModuleOp mod);

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITONXPU_ANALYSIS_TILEANALYSIS_H
