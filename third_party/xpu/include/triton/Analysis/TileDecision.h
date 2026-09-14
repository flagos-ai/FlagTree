#ifndef TRITONXPU_ANALYSIS_TILEDECISION_H
#define TRITONXPU_ANALYSIS_TILEDECISION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <string>

//===----------------------------------------------------------------------===//
// Tile decision: picks `iterNum` out of the trip counts UnrollControl can
// express, by narrowing the candidate set one tier at a time.
//
// The order is a dictionary order, not a weighted sum: a lower tier is never
// traded against a higher one, so no unit conversion between "registers" and
// "instructions" has to be invented. Tier 1 and 2 are hard -- they filter --
// and tier 3 is soft: it ranks whatever survived. Adding a consideration means
// pushing one more object into a tier, or adding a tier; nothing registers
// itself and there is no weight to tune.
//
// Every criterion has to end up in the remark (see `Decision::perTierTrace`).
// A criterion that cannot be observed cannot be verified, and the one time this
// pass shipped a silent fallback -- a mis-set budget making
// `budget-unreachable` the normal case -- the symptom was read as the design
// for weeks while every probe ran 8..11% slow.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace xpu {

// One expressible trip count. The boundary set and the segment id land here
// once M4/M5 exist; until then the candidate *is* the trip count.
struct TileCandidate {
  int64_t iterNum = 1;
};

// What the criteria are allowed to read. Filled once per decision point so the
// criteria stay free of pass state and are testable on their own.
struct TileContext {
  int64_t numCol = 1;
  int64_t widthPerCore = 1;
  // Peak simultaneously-live vector registers over the block the tile loop
  // will live in, and the widest vector row in it. Both come from
  // getRegPressure/getBlockRegPressure.
  int64_t peakVRegs = 0;
  int64_t maxVecWidth = 1;
  int64_t scalarPeak = -1; // reported only, never drives a tier
  // Narrowest vector row in slots, or 0 when the segment holds no vector
  // value. A legality bound, so it is already folded into the candidate set;
  // it is carried here only to name which ceiling a fallback hit.
  int64_t vecRow = 0;
  int64_t vrfBudget = 0;
  // iter_args the tile loop will carry: 0 at a pointwise store segment
  // (UnrollControl.cpp:1263 passes an empty range), one per reduce data
  // operand at a reduce segment (:1908).
  int64_t loopResults = 0;
  // How many times a value crosses between vector and scalar form inside the
  // segment: one per pack and one per unpack, counted separately rather than in
  // pairs because nothing guarantees they come in pairs once M4 places them.
  int64_t boundaryCrossings = 0;
  // Lanes in one hardware vector at the boundary -- 16 for f32, read off the
  // pack/unpack `vector<16xf32>` element type. NOT `maxVecWidth`, which counts
  // whole vector slots per core in elements: the two are numerically equal in
  // every geometry measured so far, which is exactly why confusing them is
  // silent. 0 when the segment has no boundary.
  int64_t vecLanes = 0;
};

// What one criterion said about the candidate set, for the remark.
struct CriterionTrace {
  llvm::StringRef name;
  unsigned tier = 0;
  int64_t candidatesIn = 0;
  int64_t candidatesOut = 0;
  std::optional<int64_t> chosenCost; // set by soft criteria only
  std::string why;                   // non-empty when the tier vetoed
  // Report-only criteria fill these instead of narrowing: the candidate they
  // would have picked, so the remark can say whether admitting them would move
  // the factor. `std::nullopt` means the criterion had no opinion at this site.
  bool reportOnly = false;
  std::optional<int64_t> shadowPick;
};

struct Decision {
  int64_t iterNum = 1;
  std::string why;
  llvm::SmallVector<CriterionTrace> perTierTrace;
  // What the soft tier would pick if the tier-1 pressure budget only reported
  // instead of filtering -- the counterfactual for the demotion §3.2.1 plans
  // ("P4 之后这一层应当降级成 Tier 3 里的一个带价项"). Report-only: nothing in
  // the pass reads it, it exists so the demotion's blast radius is measured
  // before it is done rather than after. Report-only tier-3 costs *do* count
  // towards it (they do not count towards `iterNum`), because the demoted world
  // is the one where every priced term is trusted.
  std::optional<int64_t> budgetOffPick;
};

class TileCriterion {
public:
  virtual ~TileCriterion() = default;
  virtual llvm::StringRef name() const = 0;
  // 1 = hard (VRF pressure proxy), 2 = hard (LM capacity),
  // 3 = soft (tie-break).
  virtual unsigned tier() const = 0;

  // Hard criteria implement this; fill `why` when rejecting.
  virtual bool isFeasible(const TileCandidate &, const TileContext &,
                          std::string &why) const {
    return true;
  }
  // Soft criteria implement this. Smaller is better; std::nullopt means the
  // criterion has no opinion on this candidate.
  virtual std::optional<int64_t> cost(const TileCandidate &,
                                      const TileContext &) const {
    return std::nullopt;
  }

  // A criterion that is measured but not yet trusted: its cost is computed and
  // reported, and the candidate set passes through untouched. Every new soft
  // criterion starts here, because a criterion with no case that exercises it
  // cannot be shown to be right -- and the only two magic factors this pass
  // ever shipped (`vrfBudget=4`, the two pinned unroll numbers) both got in by
  // being plausible instead of being falsified.
  virtual bool reportOnly() const { return false; }
};

// The trip count a tree of the widest row needs so that the block's vector
// pressure lands inside the calibrated budget.
//
// NOT a zero-spill criterion, and it must not be described as one: the budget
// is a pressure ceiling calibrated against measured time, and on the welford
// reduce segment the time-optimal point spills 14 accumulators while the
// spill-free point is 15.6% slower. Once the pack/unpack unit price is
// calibrated this whole criterion moves to tier 3, so nothing about it may
// assume it sits on a hard tier.
int64_t vrfBudgetTarget(const TileContext &ctx);

// `vrfBudgetTarget`'s algebra solved the other way: the per-iteration pressure
// a tree of `peak` vregs is under at trip count `k`. `peak` is explicit rather
// than read off `ctx` because step 3.4 has to ask the question about a peak the
// decision was *not* taken against -- the relaxed decision uses the tree peak
// while the register file still sees the block peak -- and because
// `pressure-price` asks it about `ctx.peakVRegs`. Keep one copy of the algebra,
// or the two answers drift the way the two copies of `target` once did.
int64_t pressureAtTrip(const TileContext &ctx, int64_t peak, int64_t k);

// The second half of `vrfBudgetTarget` on its own: convert a trip count
// computed against the *block* pressure into this tree's row. Split out so that
// a candidate pressure formula (step 3.7's "subtract the loop-invariant part
// instead of dividing it") is converted by the same algebra as the one in
// production, which is what makes the two comparable at all.
int64_t vrfBudgetTargetFrom(const TileContext &ctx, int64_t blockTarget);

// Assembles the first-version criteria and runs the narrowing. `candidates`
// must already be the expressible trip counts, ascending.
class TileDecider {
public:
  TileDecider();
  Decision decide(llvm::ArrayRef<int64_t> candidates,
                  const TileContext &ctx) const;

private:
  llvm::SmallVector<std::unique_ptr<TileCriterion>> criteria;
};

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITONXPU_ANALYSIS_TILEDECISION_H
