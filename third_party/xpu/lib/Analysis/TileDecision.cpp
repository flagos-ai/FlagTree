#include "triton/Analysis/TileDecision.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace xpu {

int64_t vrfBudgetTargetFrom(const TileContext &ctx, int64_t blockTarget) {
  // Converted to this tree's row: what is shared between the trees in one block
  // is the per-iteration width, not the number of iterations.
  int64_t target = llvm::divideCeil(blockTarget * ctx.widthPerCore,
                                    std::max<int64_t>(ctx.maxVecWidth, 1));
  return std::max<int64_t>(target, 1);
}

int64_t pressureAtTrip(const TileContext &ctx, int64_t peak, int64_t k) {
  return llvm::divideCeil(peak * std::max<int64_t>(ctx.widthPerCore, 1),
                          std::max<int64_t>(k, 1) *
                              std::max<int64_t>(ctx.maxVecWidth, 1));
}

int64_t vrfBudgetTarget(const TileContext &ctx) {
  // Trip count a tree of the widest row would need to fit the budget, ...
  int64_t blockTarget =
      ctx.vrfBudget > 0 ? llvm::divideCeil(ctx.peakVRegs, ctx.vrfBudget) : 1;
  return vrfBudgetTargetFrom(ctx, blockTarget);
}

namespace {

class VRFBudgetCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "vrf-budget"; }
  unsigned tier() const override { return 1; }

  bool isFeasible(const TileCandidate &cand, const TileContext &ctx,
                  std::string &why) const override {
    // The smallest surviving trip count is the largest tile that fits the
    // budget, so admitting everything at or above the target never over-tiles:
    // tier 3 picks the smallest one back.
    if (cand.iterNum >= vrfBudgetTarget(ctx))
      return true;
    why = "budget-unreachable";
    return false;
  }
};

// Placeholder for M3. Deciding nothing is deliberate: there is no calibrated
// capacity number yet (§3.3.1 -- legalize does not enforce an alloca ceiling
// either), and a criterion with a made-up bound would silently move decisions.
// It stays on the tier so the remark has a slot for the occupancy figure the
// moment M3 can produce one.
class LMCapacityCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "lm-capacity"; }
  unsigned tier() const override { return 2; }

  bool isFeasible(const TileCandidate &, const TileContext &,
                  std::string &) const override {
    return true;
  }
};

// Loop overhead, analytically: the index arithmetic plus a copy in and out for
// every value the loop carries, charged once per iteration. No calibration and
// no new measurement -- and being monotone in `iterNum`, its argmin is the
// smallest surviving trip count, which is exactly what this pass picked before
// the tiers existed. That is the point: the framework lands as a refactor with
// no observable change, and criteria that *do* move decisions get added one at
// a time behind their own evidence.
class LoopOverheadCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "loop-overhead"; }
  unsigned tier() const override { return 3; }

  std::optional<int64_t> cost(const TileCandidate &cand,
                              const TileContext &ctx) const override {
    return cand.iterNum * (1 + 2 * std::max<int64_t>(ctx.loopResults, 0));
  }
};

// The vector<->scalar boundary, priced in static instructions -- the unit P4
// calibrated, and the only unit this tier is allowed to use besides an analytic
// count (§2.4: instruction *totals* as a proxy for time are measured to be
// anti-correlated, but the price of one named construct is not a proxy, it is
// the construct's own cost).
//
// One crossing of N scalars at vector width W costs N scalar LM accesses, one
// vector LM access per whole or partial vector, and one LM fence:
//
//     crossing(N, W) = N + ceil(N/W) + 1
//
// so an unpack/pack pair is the `2N + 2N/W + 2` of §3.6, and at N=64 / W=16
// that is 138 instructions -- measured as ~23us on `add`, 12% of its 191us
// baseline, at the ~1ns per scalar instruction per tile-body execution P4
// fitted on two geometries (`findings.md` §1.33). Costs stay in instructions
// here; converting to time needs the occupancy, which the decider does not
// know.
//
// The crossing sits *inside* the tile loop today, so it is paid `iterNum` times
// on N = widthPerCore/iterNum scalars each:
//
//     iterNum * crossing(widthPerCore/iterNum, W)
//       = widthPerCore + iterNum * ceil(widthPerCore / (iterNum * W)) + iterNum
//
// Above the vector-width floor (widthPerCore/iterNum >= W) the middle term is
// just widthPerCore/W and tiling changes only the fence count, not the data
// movement. Below it the ceiling bites: a tile narrower than one vector still
// pays a whole vector access per iteration, so the vector side stops shrinking
// and grows with iterNum instead. Both are predictions, not assumptions, and
// the P4 probe (`TRITONXPU_PROBE_SIDE=scalar`) is the case that can falsify
// them.
//
// M4 may hoist a crossing out of the loop, at which point the repetition count
// stops being `iterNum` and this has to be told where the crossing landed
// (§3.6: unit price x repetitions, repetitions come from the materialisation
// site).
class BoundaryPriceCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "boundary-price"; }
  unsigned tier() const override { return 3; }
  // Report-only: no probe in the suite materialises a boundary today (reduce
  // region combine is default-off), so this criterion has no case that could
  // show it wrong yet. It ranks nothing until one exists.
  bool reportOnly() const override { return true; }

  std::optional<int64_t> cost(const TileCandidate &cand,
                              const TileContext &ctx) const override {
    if (ctx.boundaryCrossings <= 0 || ctx.vecLanes <= 0)
      return std::nullopt;
    int64_t w = ctx.vecLanes;
    int64_t perCore = std::max<int64_t>(ctx.widthPerCore, 1);
    int64_t iterNum = std::max<int64_t>(cand.iterNum, 1);
    int64_t vectorOps = iterNum * llvm::divideCeil(perCore, iterNum * w);
    return ctx.boundaryCrossings * (perCore + vectorOps + iterNum);
  }
};

// Pressure, priced -- the term §3.2.1's demotion needs before tier 1 may stop
// filtering. Without it the soft tier has only the monotone loop overhead left
// and collapses every pressure-bound site to `iterNum=1` (`findings.md` §1.36).
//
// The form comes from §1.20's attribution, not from counting spills: welford's
// `iterNum=1` pays +105us over `iterNum=2` while vector spill/reload accounts
// for 3.5% of it. What grows is the *lowering of the same IR* -- 2x lane
// extract, 2x mask move, 2x index arithmetic, 1.5x scalar divide. So the price
// is "predicted over-budget vregs x instructions each one costs", and
// `spills(k)` is deliberately absent (§2.4 rejected `spillPrice x spills(k)`).
//
//   over(k)  = max(0, ceil(peakVRegs * widthPerCore / (k * maxVecWidth))
//                      - vrfBudget)
//   price(k) = kPressureInstrsPerVReg * over(k)
//
// The inner expression is `vrfBudgetTarget`'s algebra solved for the pressure
// at a given trip count, on purpose: the demotion has to be an *identity* on
// today's decisions before it can be an improvement, and sharing the algebra is
// what makes that checkable instead of coincidental. The
// `widthPerCore / maxVecWidth` factor is the same block-to-tree conversion
// (§1.10) -- `peakVRegs` is the block's peak at its widest row, so a narrower
// tree in the same block is not under `peakVRegs/k` of pressure.
//
// Calibration: welford region-reduce has `peakVRegs=32` and `budget=24`, so
// over(1)=8 and over(2)=0 against a measured +578 machine instructions per row
// => ~72 instructions per over-budget vreg per row (`findings.md` §1.38;
// §1.20's +720 and the 90 that followed from it came from a trip count that
// charged every inner loop a welford-shaped guess, corrected in §1.38). That is
// one pair on one probe, which is why this is report-only -- and the one other
// candidate pair, layernorm at `vrfBudget` 48 vs 24, cannot confirm it: over(k)
// is 0 on both sides there while the code still grows +76 instructions per row,
// so the budget-derived pressure is blind at that point (§1.38). The 13-site
// replay checks the penalty's *support* -- where it is zero and where it is not
// -- and cannot pin the constant down: at every site today over(k) is either 0
// or large enough that any positive constant yields the same argmin.
//
// A second point does exist -- vary k instead of the budget, which is what the
// welford calibration itself did -- and it *refutes* a single constant: over
// the pinned iterNum=1/2 pair the same fit reads 72.2 on welford, 0.96 on
// layernorm and 1.86 on mixedwidth, while softmax grows +211 instructions per
// row with over(k) identically 0 (`findings.md` §1.39). The constant stays at
// 72 because this term does not enter the decision sum and swapping in 0.96 or
// a mean would only trade one unsupported constant for another; what the
// measurement rules out is the *form*, so promoting this criterion now needs a
// different form, not another calibration point -- and tier 1 therefore still
// may not be demoted.
//
// §1.40 closed the last candidate for such a form. Decomposing that growth per
// instruction class shows the lane-extract / mask-move / scalar-divide
// signature §1.20 attributed to pressure does not order with pressure at all:
// it is exactly zero on the two probes with the largest over(k) (layernorm 24,
// mixedwidth 48) and large on the two with the smallest (welford 8, softmax 0),
// which no monotone function of peakVRegs/k/vrfBudget can fit. The only class
// that does respond is spill/reload -- the term the hardware already rejected
// (§1.17, and §2.4 rejected `spillPrice x spills(k)`) -- and even its support
// is wrong: welford still spills where over(k)=0. So this ratio supports a
// feasibility bound, not a price.
//
// It is nevertheless summed into tier 3 rather than left report-only, and the
// reason it is safe is an identity, not an experiment. Tier 1 admits exactly
// `k >= vrfBudgetTarget(ctx)`, and for every such k:
//
//   k * maxVecWidth / widthPerCore >= blockTarget >= peakVRegs / vrfBudget
//     =>  peakVRegs * widthPerCore / (k * maxVecWidth) <= vrfBudget
//     =>  over(k) = 0
//
// so the price is identically zero on the whole feasible set and cannot move
// `iterNum` while tier 1 filters -- which is what the 13-site replay shows
// (`cost=0` and `budget-off pick ... moved=no` at every site) and what keeps
// the golden IR byte-identical. The term is here so the sum has the shape the
// demotion needs, not because it is trusted: the moment tier 1 stops filtering,
// `kPressureInstrsPerVReg` starts driving decisions, and §1.39 measured it to
// be 72.2 / 0.96 / 1.86 on three probes, i.e. not a constant. **Demoting tier 1
// therefore still needs a new observable** -- a real-machine time sweep of
// pressure, or a pressure measure not derived from peakVRegs and vrfBudget --
// and `budget-off pick` remains the port that says what that demotion would
// cost per site.
constexpr int64_t kPressureInstrsPerVReg = 72;

class PressurePriceCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "pressure-price"; }
  unsigned tier() const override { return 3; }

  std::optional<int64_t> cost(const TileCandidate &cand,
                              const TileContext &ctx) const override {
    if (ctx.peakVRegs <= 0 || ctx.vrfBudget <= 0)
      return std::nullopt;
    int64_t perIter = pressureAtTrip(ctx, ctx.peakVRegs, cand.iterNum);
    int64_t over = perIter - ctx.vrfBudget;
    return over > 0 ? kPressureInstrsPerVReg * over : 0;
  }
};

} // namespace

// The counterfactual for §3.2.1's demotion plan: tier 1 stops filtering and
// only prices, so the soft tier ranks the whole legal set. Evaluated rather
// than derived on purpose -- today's tier-3 cost happens to be monotone in
// `iterNum`, which makes the answer "the smallest legal factor" by arithmetic,
// but that is a property of the current cost, not of the design, and the
// demotion's whole point is to add a term that is *not* monotone.
//
// Report-only tier-3 criteria *are* summed here, unlike in `decide()`: this
// answers "what would the demoted world pick", and in that world every priced
// term counts. `decide()` still ignores them, so nothing observable moves --
// the two differ exactly by the terms that are measured but not yet trusted.
static std::optional<int64_t>
pickWithoutTier1(llvm::ArrayRef<int64_t> candidates, const TileContext &ctx,
                 llvm::ArrayRef<std::unique_ptr<TileCriterion>> criteria) {
  llvm::SmallVector<int64_t> live(candidates.begin(), candidates.end());
  for (auto &c : criteria) {
    if (c->tier() < 2 || c->reportOnly())
      continue;
    llvm::SmallVector<int64_t> kept;
    for (int64_t k : live) {
      std::string why;
      if (c->isFeasible(TileCandidate{k}, ctx, why))
        kept.emplace_back(k);
    }
    if (kept.empty())
      return std::nullopt;
    live = std::move(kept);
  }
  std::optional<int64_t> best, bestK;
  for (int64_t k : live) {
    std::optional<int64_t> sum;
    for (auto &c : criteria) {
      if (c->tier() < 3)
        continue;
      if (auto c0 = c->cost(TileCandidate{k}, ctx))
        sum = sum.value_or(0) + *c0;
    }
    if (!sum)
      continue;
    if (!best || *sum < *best) {
      best = sum;
      bestK = k;
    }
  }
  return bestK;
}

TileDecider::TileDecider() {
  criteria.emplace_back(std::make_unique<VRFBudgetCriterion>());
  criteria.emplace_back(std::make_unique<LMCapacityCriterion>());
  criteria.emplace_back(std::make_unique<LoopOverheadCriterion>());
  criteria.emplace_back(std::make_unique<BoundaryPriceCriterion>());
  criteria.emplace_back(std::make_unique<PressurePriceCriterion>());
}

Decision TileDecider::decide(llvm::ArrayRef<int64_t> candidates,
                             const TileContext &ctx) const {
  Decision d;
  if (candidates.empty()) {
    d.iterNum = 1;
    d.why = "no-legal-trip-count";
    return d;
  }
  // The largest expressible trip count is the fallback: when a tier admits
  // nothing, the most tiled legal point is the closest thing to satisfying it.
  int64_t maxLegal = candidates.back();

  // Computed before the narrowing so the veto path carries it too -- a site
  // where the budget vetoed everything is exactly where the demotion would
  // change the most.
  d.budgetOffPick = pickWithoutTier1(candidates, ctx, criteria);

  llvm::SmallVector<int64_t> live(candidates.begin(), candidates.end());
  unsigned maxTier = 0;
  for (auto &c : criteria)
    maxTier = std::max(maxTier, c->tier());

  for (unsigned tier = 1; tier <= maxTier; ++tier) {
    // Prices inside one tier add up and the argmin is taken once, after every
    // criterion in the tier has spoken. This loop used to collapse `live` to
    // the winner after each soft criterion, which made the *second* soft
    // criterion in a tier structurally unable to say anything: it only ever saw
    // the single candidate the first one had left. `pickWithoutTier1` summed
    // all along, so the counterfactual and the decision disagreed about what
    // "tier 3" meant -- and the disagreement was invisible while
    // `loop-overhead` was the only trusted price.
    llvm::SmallDenseMap<int64_t, int64_t, 8> tierCost;
    llvm::SmallVector<std::pair<size_t, const TileCriterion *>, 4> pricedTraces;
    bool anyCost = false;
    for (auto &c : criteria) {
      if (c->tier() != tier)
        continue;
      CriterionTrace trace;
      trace.name = c->name();
      trace.tier = tier;
      trace.candidatesIn = live.size();

      // Report-only criteria are priced and traced, but the set walks past them
      // untouched -- including `d.iterNum`, which must stay bit-for-bit what it
      // was before the criterion existed.
      if (c->reportOnly()) {
        trace.reportOnly = true;
        trace.candidatesOut = live.size();
        std::optional<int64_t> best;
        for (int64_t k : live) {
          auto c0 = c->cost(TileCandidate{k}, ctx);
          if (!c0)
            continue;
          if (!best || *c0 < *best) {
            best = c0;
            trace.shadowPick = k;
          }
        }
        trace.chosenCost = best;
        d.perTierTrace.emplace_back(std::move(trace));
        continue;
      }

      llvm::SmallVector<int64_t> kept;
      std::string why;
      for (int64_t k : live) {
        std::string thisWhy;
        if (c->isFeasible(TileCandidate{k}, ctx, thisWhy))
          kept.emplace_back(k);
        else if (why.empty())
          why = thisWhy;
      }
      if (kept.empty()) {
        // Never silently: the fallback is reported, and it is the fallback the
        // caller sees in `iterNum`.
        trace.candidatesOut = 0;
        // Which ceiling was hit is the useful part. The vector row is a
        // correctness bound the model cannot trade away; the scalar row is one
        // that tiling could in principle widen.
        trace.why = ctx.vecRow && maxLegal == ctx.vecRow
                        ? "vector-row-bound"
                        : (why.empty() ? "infeasible" : why);
        d.why = trace.why;
        d.iterNum = maxLegal;
        d.perTierTrace.emplace_back(std::move(trace));
        return d;
      }
      live = std::move(kept);
      trace.candidatesOut = live.size();

      // Soft side: accumulate this criterion's price per candidate. A criterion
      // with no opinion on a candidate contributes nothing to that candidate's
      // sum, which is the same rule `pickWithoutTier1` uses.
      bool thisPrices = false;
      for (int64_t k : live)
        if (auto c0 = c->cost(TileCandidate{k}, ctx)) {
          tierCost[k] += *c0;
          thisPrices = true;
        }
      if (thisPrices) {
        anyCost = true;
        pricedTraces.emplace_back(d.perTierTrace.size(), c.get());
      }
      d.perTierTrace.emplace_back(std::move(trace));
    }
    if (!anyCost)
      continue;
    int64_t nBefore = live.size();
    std::optional<int64_t> best;
    int64_t bestK = live.front();
    for (int64_t k : live) {
      auto it = tierCost.find(k);
      if (it == tierCost.end())
        continue;
      if (!best || it->second < *best) {
        best = it->second;
        bestK = k;
      }
    }
    live.assign(1, bestK);
    // Each priced criterion reports its own share *at the winner*, so the
    // bracket groups add up to `cost-sum` instead of each showing its own
    // minimum at a possibly different candidate.
    for (auto &[idx, c] : pricedTraces)
      d.perTierTrace[idx].chosenCost = c->cost(TileCandidate{bestK}, ctx);
    CriterionTrace sum;
    sum.name = "cost-sum";
    sum.tier = tier;
    sum.candidatesIn = nBefore;
    sum.candidatesOut = 1;
    sum.chosenCost = best;
    d.perTierTrace.emplace_back(std::move(sum));
  }
  d.iterNum = live.front();
  if (d.why.empty())
    d.why = "budget";
  return d;
}

} // namespace xpu
} // namespace triton
} // namespace mlir
